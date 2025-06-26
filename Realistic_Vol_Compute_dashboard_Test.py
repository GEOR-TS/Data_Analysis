import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Tuple, Dict, Optional


# Set page config
st.set_page_config(
    page_title="Volatility Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #c9d3df;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)




class VolatilityAnalyzer:
    """
    A class to compute various volatility measures for ETHUSDC trading data with Plotly visualizations.
    """

    def __init__(self, parquet_file: str):
        """Initialize with the parquet file path."""
        self.df = pd.read_parquet(parquet_file)
        self.df = self.df.sort_values('timestamp')

    def get_date_range(self):
        """Get the available date range in the data."""
        return self.df['timestamp'].min(), self.df['timestamp'].max()

    def get_price_data(self, start_date: str, end_date: str, frequency: str = '5min') -> pd.DataFrame:
        """
        Get price data for the specified date range and resample to given frequency.
        """
        # Convert string dates to timezone-aware timestamps
        start = pd.to_datetime(start_date).tz_localize('UTC')
        end = pd.to_datetime(end_date).tz_localize('UTC')

        # Filter data
        mask = (self.df['timestamp'] >= start) & (self.df['timestamp'] <= end)
        filtered_df = self.df[mask].copy()

        if len(filtered_df) == 0:
            raise ValueError(f"No data found between {start} and {end}")

        # Set timestamp as index
        filtered_df.set_index('timestamp', inplace=True)

        # Resample to create OHLC data
        ohlc = filtered_df['price'].resample(frequency).ohlc()
        volume = filtered_df['volume'].resample(frequency).sum()
        vwap = (filtered_df['price'] * filtered_df['volume']).resample(frequency).sum() / volume

        # Combine into single dataframe
        resampled = pd.DataFrame({
            'open': ohlc['open'],
            'high': ohlc['high'],
            'low': ohlc['low'],
            'close': ohlc['close'],
            'volume': volume,
            'vwap': vwap
        })

        # Remove periods with no trades
        resampled = resampled.dropna()

        # Calculate returns
        resampled['returns'] = resampled['close'].pct_change()
        resampled['log_returns'] = np.log(resampled['close'] / resampled['close'].shift(1))

        return resampled

    def calculate_volatility(self, start_date: str, end_date: str,
                             frequency: str = '5min',
                             method: str = 'close') -> Dict[str, float]:
        """
        Calculate various volatility measures for the specified period.
        """
        # Get resampled data
        data = self.get_price_data(start_date, end_date, frequency)

        # Determine annualization factor based on frequency
        annualization_factors = {
            '1min': 525600,
            '5min': 105120,
            '15min': 35040,
            '30min': 17520,
            '1h': 8760,
            '4h': 2190,
            '1D': 365,
        }

        ann_factor = annualization_factors.get(frequency, 365)

        results = {}

        # 1. Close-to-Close Volatility (Traditional)
        if method in ['all', 'close']:
            close_vol = data['log_returns'].std() * np.sqrt(ann_factor)
            results['close_to_close'] = close_vol * 100

        # 2. Parkinson Volatility (High-Low)
        if method in ['all', 'parkinson']:
            hl_ratio = np.log(data['high'] / data['low'])
            parkinson_vol = np.sqrt(1 / (4 * np.log(2)) * (hl_ratio ** 2).mean()) * np.sqrt(ann_factor)
            results['parkinson'] = parkinson_vol * 100

        # 3. Garman-Klass Volatility
        if method in ['all', 'garman_klass']:
            hl_ratio = np.log(data['high'] / data['low'])
            co_ratio = np.log(data['close'] / data['open'])
            gk_vol = np.sqrt(
                (0.5 * (hl_ratio ** 2) - (2 * np.log(2) - 1) * (co_ratio ** 2)).mean()
            ) * np.sqrt(ann_factor)
            results['garman_klass'] = gk_vol * 100

        # 4. Realized Volatility
        if method in ['all', 'realized']:
            # Get all trades in the period
            start = pd.to_datetime(start_date).tz_localize('UTC')
            end = pd.to_datetime(end_date).tz_localize('UTC')
            mask = (self.df['timestamp'] >= start) & (self.df['timestamp'] <= end)
            trades = self.df[mask].copy()

            if len(trades) > 1:
                # Calculate trade-to-trade returns
                trades['log_returns'] = np.log(trades['price'] / trades['price'].shift(1))

                # Calculate realized volatility
                avg_time_between_trades = trades['timestamp'].diff().dt.total_seconds().mean()
                trades_per_year = 365 * 24 * 60 * 60 / avg_time_between_trades

                realized_vol = trades['log_returns'].std() * np.sqrt(trades_per_year)
                results['realized'] = realized_vol * 100

        return results

    def plot_volatility_cone(self, lookback_days: int = 30,
                             frequencies: list = ['1h', '4h', '1D']) -> go.Figure:
        """
        Plot an interactive volatility cone showing volatility distribution over time.
        """
        end_date = self.df['timestamp'].max()

        # Create subplots
        fig = make_subplots(
            rows=len(frequencies),
            cols=1,
            subplot_titles=[f'Rolling {lookback_days}-day Volatility ({freq} frequency)'
                            for freq in frequencies],
            vertical_spacing=0.1
        )

        colors = ['blue', 'green', 'red', 'purple', 'orange']

        for idx, freq in enumerate(frequencies):
            volatilities = []
            dates = []

            # Calculate rolling volatility
            current_date = end_date - timedelta(days=lookback_days * 2)
            while current_date <= end_date - timedelta(days=lookback_days):
                try:
                    start = current_date.strftime('%Y-%m-%d')
                    end = (current_date + timedelta(days=lookback_days)).strftime('%Y-%m-%d')

                    vols = self.calculate_volatility(start, end, freq, method='close')
                    volatilities.append(vols['close_to_close'])
                    dates.append(current_date)
                except:
                    pass

                current_date += timedelta(days=1)

            if volatilities:
                # Add line trace
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=volatilities,
                        mode='lines',
                        name=f'{freq} volatility',
                        line=dict(color=colors[idx % len(colors)], width=2),
                        fill='tozeroy'
                    ),
                    row=idx + 1, col=1
                )

                # Add average line
                avg_vol = np.mean(volatilities)
                fig.add_hline(
                    y=avg_vol,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"Avg: {avg_vol:.1f}%",
                    annotation_position="right",
                    row=idx + 1, col=1
                )

                # Update y-axis
                fig.update_yaxes(
                    title_text="Annualized Volatility (%)",
                    row=idx + 1, col=1
                )

        # Update layout
        fig.update_layout(
            height=300 * len(frequencies),
            showlegend=True,
            title_text=f"Volatility Cone Analysis - {lookback_days}-Day Rolling Window",
            hovermode='x unified'
        )

        fig.update_xaxes(title_text="Date", row=len(frequencies), col=1)

        return fig

    def plot_intraday_volatility_pattern(self, start_date: str, end_date: str) -> go.Figure:
        """
        Analyze and plot hourly volatility patterns by hour of day.
        """
        # Get hourly data
        data = self.get_price_data(start_date, end_date, '1h')

        # Add hour of day
        data['hour'] = data.index.hour

        # Calculate volatility by hour
        hourly_vol = data.groupby('hour')['log_returns'].agg(['std', 'count'])
        hourly_vol['annualized_vol'] = hourly_vol['std'] * np.sqrt(8760) * 100

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Hourly Volatility Pattern', 'Number of Hourly Periods by Hour'),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )

        # Add volatility bar chart
        fig.add_trace(
            go.Bar(
                x=hourly_vol.index,
                y=hourly_vol['annualized_vol'],
                name='Volatility',
                marker_color='lightblue',
                text=hourly_vol['annualized_vol'].round(1),
                textposition='outside',
                hovertemplate='Hour: %{x}<br>Volatility: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # Add average line
        avg_vol = hourly_vol['annualized_vol'].mean()
        fig.add_hline(
            y=avg_vol,
            line_dash="dash",
            line_color="red",
            annotation_text=f"24h Average: {avg_vol:.1f}%",
            annotation_position="right",
            row=1, col=1
        )

        # Add count bar chart
        fig.add_trace(
            go.Bar(
                x=hourly_vol.index,
                y=hourly_vol['count'],
                name='Count',
                marker_color='lightgreen',
                text=hourly_vol['count'],
                textposition='outside',
                hovertemplate='Hour: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=1
        )

        # Update axes
        fig.update_xaxes(title_text="Hour of Day (UTC)", row=2, col=1)
        fig.update_yaxes(title_text="Annualized Volatility (%)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)

        # Update layout
        fig.update_layout(
            height=700,
            showlegend=False,
            title_text=f"Hourly Volatility Analysis ({start_date} to {end_date})",
            hovermode='x'
        )

        return fig

    def plot_volatility_heatmap(self, start_date: str, end_date: str,
                                frequency: str = '1h') -> go.Figure:
        """
        Create a heatmap of volatility by day of week and hour.
        """
        # Get data
        data = self.get_price_data(start_date, end_date, frequency)

        # Calculate rolling volatility
        window = 24 if frequency == '1h' else 6 if frequency == '4h' else 1
        data['rolling_vol'] = data['log_returns'].rolling(window=window).std() * np.sqrt(8760) * 100

        # Add time components
        data['hour'] = data.index.hour
        data['dayofweek'] = data.index.dayofweek

        # Create pivot table
        pivot = data.pivot_table(
            values='rolling_vol',
            index='hour',
            columns='dayofweek',
            aggfunc='mean'
        )

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            y=pivot.index,
            colorscale='RdYlBu_r',
            text=np.round(pivot.values, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hovertemplate='%{x}<br>Hour: %{y}<br>Avg Vol: %{z:.1f}%<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title=f'Average Volatility Heatmap by Day and Hour',
            xaxis_title='Day of Week',
            yaxis_title='Hour of Day (UTC)',
            height=600
        )

        return fig

    def plot_volatility_comparison(self, periods: list, frequency: str = '1h') -> go.Figure:
        """
        Compare volatility across different time periods.
        """
        fig = go.Figure()

        colors = px.colors.qualitative.Set3

        for idx, (start, end, label) in enumerate(periods):
            try:
                vols = self.calculate_volatility(start, end, frequency, method='all')

                # Create bar chart data
                methods = list(vols.keys())
                values = list(vols.values())

                fig.add_trace(
                    go.Bar(
                        name=label,
                        x=methods,
                        y=values,
                        text=[f'{v:.1f}%' for v in values],
                        textposition='outside',
                        marker_color=colors[idx % len(colors)],
                        hovertemplate='%{x}<br>%{y:.1f}%<extra></extra>'
                    )
                )
            except Exception as e:
                st.error(f"Error for period {label}: {e}")

        # Update layout
        fig.update_layout(
            title='Volatility Comparison Across Different Periods',
            xaxis_title='Volatility Method',
            yaxis_title='Annualized Volatility (%)',
            barmode='group',
            height=500,
            hovermode='x'
        )

        # Clean up x-axis labels
        if methods:
            fig.update_xaxes(
                ticktext=[m.replace('_', ' ').title() for m in methods],
                tickvals=methods
            )

        return fig


# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
    st.session_state.data_loaded = False


# Main app
def main():
    
    st.markdown("""
        <style>
            /* Custom styling for the header */
            .main > div:first-child {
                padding-top: 0rem;
            }

            .header-container {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 2rem;
            }

        </style>
        """, unsafe_allow_html=True)

    # Create header with logo
    header_col1, header_col2, header_col3 = st.columns([1, 2, 1])

    st.title("📊 ETH/USDC Volatility Analysis Dashboard")
    st.markdown("---")

    # Initialize analyzer if not already in session state
    if 'analyzer' not in st.session_state or st.session_state.analyzer is None:
        try:
            st.session_state.analyzer = VolatilityAnalyzer('Kraken-ETHUSDC-2024-2025.parquet')
            st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"Error loading data file: {str(e)}")
            st.error("Please ensure 'Kraken-ETHUSDC-2024-2025.parquet' is in the current directory.")
            st.stop()

    # Verify analyzer is loaded
    if st.session_state.analyzer is None:
        st.error("Failed to initialize data analyzer.")
        st.stop()

    st.markdown("""
    <style>
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #ffffff !important;
        }

        /* Sidebar content area */
        section[data-testid="stSidebar"] > div {
            background-color: #ffffff !important;
        }

        /* Remove any gradient or pattern */
        section[data-testid="stSidebar"] > div:first-child {
            background-color: #ffffff !important;
            background-image: none !important;
        }

        /* Style sidebar elements */
        section[data-testid="stSidebar"] .element-container {
            background-color: #ffffff !important;
        }

        /* Style input fields in sidebar to match */
        section[data-testid="stSidebar"] .stSelectbox > div,
        section[data-testid="stSidebar"] .stDateInput > div,
        section[data-testid="stSidebar"] .stNumberInput > div {
            background-color: #d7d7d9  !important;
        }

        /* Style the sidebar header text */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #1a1a1a !important;
        }

        /* Remove sidebar border */
        section[data-testid="stSidebar"] > div > div {
            border: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar for configuration
    with st.sidebar:

        st.header("⚙️ Configuration")

        try:
            min_date, max_date = st.session_state.analyzer.get_date_range()
        except Exception as e:
            st.error(f"Error getting date range: {str(e)}")
            st.stop()

        st.markdown("### 📅 Date Range Selection")

        # Date inputs
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=max_date.date() - timedelta(days=30),
                min_value=min_date.date(),
                max_value=max_date.date()
            )

        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date.date(),
                min_value=min_date.date(),
                max_value=max_date.date()
            )

        # Frequency selection
        st.markdown("### ⏱️ Time Frequency")
        frequency = st.selectbox(
            "Select Frequency",
            options=['1min', '5min', '15min', '30min', '1h', '4h', '1D'],
            index=4  # Default to 1h
        )

        st.markdown("---")
        st.markdown("### ℹ️ Data Info")
        st.info(f"**Total Records:** {len(st.session_state.analyzer.df):,}")
        st.info(f"**Date Range:** {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

    # Main content area
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Volatility Metrics",
        "📊 Hourly Patterns",
        "🗓️ Volatility Heatmap",
        "📉 Volatility Cone",
        "🔄 Period Comparison"
    ])

    # Tab 1: Volatility Metrics
    with tab1:
        st.header("Volatility Metrics Analysis")

        try:
            # Calculate volatilities
            vols = st.session_state.analyzer.calculate_volatility(
                str(start_date), str(end_date), frequency, method='all'
            )

            # Display metrics
            st.markdown("### 📊 Calculated Volatilities")
            cols = st.columns(len(vols))

            for idx, (method, value) in enumerate(vols.items()):
                with cols[idx]:
                    st.metric(
                        label=method.replace('_', ' ').title(),
                        value=f"{value:.2f}%",
                        delta=None
                    )

            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(vols.keys()),
                    y=list(vols.values()),
                    text=[f'{v:.1f}%' for v in vols.values()],
                    textposition='outside',
                    marker_color='lightblue'
                )
            ])

            fig.update_layout(
                title=f'Volatility Measures ({frequency} frequency)',
                xaxis_title='Method',
                yaxis_title='Annualized Volatility (%)',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Additional statistics
            with st.expander("📋 Additional Statistics"):
                data = st.session_state.analyzer.get_price_data(
                    str(start_date), str(end_date), frequency
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Price", f"${data['close'].mean():.2f}")
                    st.metric("Min Price", f"${data['low'].min():.2f}")
                with col2:
                    st.metric("Max Price", f"${data['high'].max():.2f}")
                    st.metric("Price Range", f"${data['high'].max() - data['low'].min():.2f}")
                with col3:
                    st.metric("Total Volume", f"{data['volume'].sum():,.2f}")
                    st.metric("Avg Daily Volume",
                              f"{data['volume'].sum() / ((end_date - start_date).days + 1):,.2f}")

        except Exception as e:
            st.error(f"Error calculating volatility: {str(e)}")

    # Tab 2: Hourly Patterns
    with tab2:
        st.header("Hourly Volatility Patterns")

        try:
            fig = st.session_state.analyzer.plot_intraday_volatility_pattern(
                str(start_date), str(end_date)
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("💡 Interpretation Guide"):
                st.markdown("""
                - **Higher bars** indicate hours with typically higher volatility
                - **UTC timezone** is used for all timestamps
                - The red dashed line shows the 24-hour average volatility
                - Bottom chart shows data availability for each hour
                """)

        except Exception as e:
            st.error(f"Error creating hourly pattern: {str(e)}")

    # Tab 3: Volatility Heatmap
    with tab3:
        st.header("Volatility Heatmap by Day and Hour")

        try:
            heatmap_freq = st.selectbox(
                "Select frequency for heatmap",
                options=['1h', '4h'],
                index=0,
                key='heatmap_freq'
            )

            fig = st.session_state.analyzer.plot_volatility_heatmap(
                str(start_date), str(end_date), heatmap_freq
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("💡 Interpretation Guide"):
                st.markdown("""
                - **Darker red** areas indicate higher volatility periods
                - **Darker blue** areas indicate lower volatility periods
                - Patterns may reveal regular market behaviors
                - All times are in UTC
                """)

        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")

    # Tab 4: Volatility Cone
    with tab4:
        st.header("Rolling Volatility Analysis")

        col1, col2 = st.columns(2)
        with col1:
            lookback_days = st.slider(
                "Lookback Period (days)",
                min_value=7,
                max_value=90,
                value=30,
                step=1
            )

        with col2:
            cone_frequencies = st.multiselect(
                "Select frequencies",
                options=['1h', '4h', '1D'],
                default=['1h', '4h', '1D']
            )

        if cone_frequencies:
            try:
                fig = st.session_state.analyzer.plot_volatility_cone(
                    lookback_days, cone_frequencies
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error creating volatility cone: {str(e)}")
        else:
            st.warning("Please select at least one frequency")

    # Tab 5: Period Comparison
    with tab5:
        st.header("Compare Volatility Across Periods")

        st.markdown("### Define Comparison Periods")

        num_periods = st.number_input(
            "Number of periods to compare",
            min_value=2,
            max_value=6,
            value=3,
            step=1
        )

        periods = []
        cols = st.columns(num_periods)

        for i in range(num_periods):
            with cols[i]:
                st.markdown(f"**Period {i + 1}**")
                p_start = st.date_input(
                    f"Start",
                    value=start_date - timedelta(days=30 * i),
                    key=f"p_start_{i}"
                )
                p_end = st.date_input(
                    f"End",
                    value=start_date - timedelta(days=30 * i - 29),
                    key=f"p_end_{i}"
                )
                p_label = st.text_input(
                    f"Label",
                    value=f"Period {i + 1}",
                    key=f"p_label_{i}"
                )
                periods.append((str(p_start), str(p_end), p_label))

        comp_freq = st.selectbox(
            "Frequency for comparison",
            options=['1h', '4h', '1D'],
            index=0,
            key='comp_freq'
        )

        if st.button("Generate Comparison"):
            try:
                fig = st.session_state.analyzer.plot_volatility_comparison(
                    periods, comp_freq
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error creating comparison: {str(e)}")


if __name__ == "__main__":
    main()


