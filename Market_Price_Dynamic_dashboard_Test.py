import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
from datetime import datetime, timedelta


warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Price Dynamics Analyzer",
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






class PriceDynamicsVisualizer:
    def __init__(self, data_path):
        """Initialize with trade data."""
        self.df = pd.read_parquet(data_path)

        # Ensure timestamp is datetime with timezone
        if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

    def plot_candlestick_chart(self, start_date=None, end_date=None, resample='5T'):
        """Create interactive candlestick chart with volume."""
        # Filter data with timezone-aware comparison
        data = self.df.copy()
        if start_date:
            start_dt = pd.to_datetime(start_date).tz_localize('UTC')
            data = data[data['timestamp'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date).tz_localize('UTC')
            data = data[data['timestamp'] <= end_dt]

        # Resample to create OHLCV data
        ohlc = data.set_index('timestamp')['price'].resample(resample).ohlc()
        volume = data.set_index('timestamp')['volume'].resample(resample).sum()

        # Combine OHLC and volume
        candlestick_data = pd.concat([ohlc, volume], axis=1)
        candlestick_data = candlestick_data.dropna()

        # Calculate moving averages
        candlestick_data['MA20'] = candlestick_data['close'].rolling(window=20).mean()
        candlestick_data['MA50'] = candlestick_data['close'].rolling(window=50).mean()

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'ETH/USDC Candlestick Chart ({resample} intervals)', 'Trading Volume')
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=candlestick_data.index,
                open=candlestick_data['open'],
                high=candlestick_data['high'],
                low=candlestick_data['low'],
                close=candlestick_data['close'],
                name='OHLC',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )

        # Add moving averages
        fig.add_trace(
            go.Scatter(
                x=candlestick_data.index,
                y=candlestick_data['MA20'],
                mode='lines',
                name='MA20',
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=candlestick_data.index,
                y=candlestick_data['MA50'],
                mode='lines',
                name='MA50',
                line=dict(color='orange', width=1.5)
            ),
            row=1, col=1
        )

        # Add volume bars
        colors = ['green' if close >= open_ else 'red'
                  for close, open_ in zip(candlestick_data['close'], candlestick_data['open'])]

        fig.add_trace(
            go.Bar(
                x=candlestick_data.index,
                y=candlestick_data['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7,
                showlegend=False
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text='ETH/USDC Price Analysis',
                font=dict(size=16, family='Arial Black')
            ),
            xaxis_rangeslider_visible=False,
            height=800,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price (USDC)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig

    def plot_price_heatmap(self, start_date=None, end_date=None):
        """Create interactive heatmap showing price patterns by hour and day of week."""
        # Filter data
        data = self.df.copy()
        if start_date:
            start_dt = pd.to_datetime(start_date).tz_localize('UTC')
            data = data[data['timestamp'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date).tz_localize('UTC')
            data = data[data['timestamp'] <= end_dt]

        # Extract hour and day of week
        data['hour'] = data['timestamp'].dt.hour
        data['dayofweek'] = data['timestamp'].dt.dayofweek
        data['day_name'] = data['timestamp'].dt.day_name()

        # Calculate average price by hour and day
        price_matrix = data.pivot_table(
            values='price',
            index='hour',
            columns='dayofweek',
            aggfunc='mean'
        )

        # Calculate volume matrix
        volume_matrix = data.pivot_table(
            values='volume',
            index='hour',
            columns='dayofweek',
            aggfunc='sum'
        )

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Average Price by Hour and Day',
                            'Trading Volume by Hour and Day (Log Scale)'),
            horizontal_spacing=0.1
        )

        # Price heatmap
        fig.add_trace(
            go.Heatmap(
                z=price_matrix.values,
                x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                y=price_matrix.index,
                colorscale='RdYlGn',
                text=np.round(price_matrix.values, 0),
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='%{x}<br>Hour: %{y}<br>Avg Price: $%{z:.2f}<extra></extra>',
                colorbar=dict(title='Avg Price (USDC)', x=0.45)
            ),
            row=1, col=1
        )

        # Volume heatmap (log scale)
        fig.add_trace(
            go.Heatmap(
                z=np.log10(volume_matrix.values + 1),
                x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                y=volume_matrix.index,
                colorscale='Blues',
                hovertemplate='%{x}<br>Hour: %{y}<br>Volume: %{customdata:,.0f}<extra></extra>',
                customdata=volume_matrix.values,
                colorbar=dict(title='Log10(Volume)', x=1.02)
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text='Price and Volume Patterns Analysis',
                font=dict(size=16, family='Arial Black')
            ),
            height=600
        )

        # Update axes
        fig.update_xaxes(title_text="Day of Week", row=1, col=1)
        fig.update_xaxes(title_text="Day of Week", row=1, col=2)
        fig.update_yaxes(title_text="Hour of Day", row=1, col=1)
        fig.update_yaxes(title_text="Hour of Day", row=1, col=2)

        return fig

    def plot_price_distribution(self, start_date=None, end_date=None):
        """Analyze price and returns distribution with interactive plots."""
        # Filter data
        data = self.df.copy()
        if start_date:
            start_dt = pd.to_datetime(start_date).tz_localize('UTC')
            data = data[data['timestamp'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date).tz_localize('UTC')
            data = data[data['timestamp'] <= end_dt]

        # Calculate returns
        data = data.sort_values('timestamp')
        data['returns'] = data['price'].pct_change()
        data['log_returns'] = np.log(data['price'] / data['price'].shift(1))

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Distribution', 'Returns Distribution',
                            'Q-Q Plot (Returns vs Normal)', 'Price vs Volume'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 1. Price distribution
        price_hist = go.Histogram(
            x=data['price'],
            nbinsx=50,
            name='Price',
            marker_color='blue',
            opacity=0.7,
            hovertemplate='Price: $%{x}<br>Count: %{y}<extra></extra>'
        )
        fig.add_trace(price_hist, row=1, col=1)

        # Add mean and median lines
        fig.add_vline(
            x=data['price'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${data['price'].mean():.2f}",
            row=1, col=1
        )
        fig.add_vline(
            x=data['price'].median(),
            line_dash="dash",
            line_color="green",
            annotation_text=f"Median: ${data['price'].median():.2f}",
            row=1, col=1
        )

        # 2. Returns distribution with normal overlay
        returns_clean = data['returns'].dropna()

        # Create histogram
        returns_hist = go.Histogram(
            x=returns_clean,
            nbinsx=100,
            name='Returns',
            marker_color='green',
            opacity=0.7,
            histnorm='probability density',
            hovertemplate='Return: %{x:.4f}<br>Density: %{y:.4f}<extra></extra>'
        )
        fig.add_trace(returns_hist, row=1, col=2)

        # Fit normal distribution
        mu, sigma = returns_clean.mean(), returns_clean.std()
        x_range = np.linspace(returns_clean.min(), returns_clean.max(), 100)
        normal_dist = stats.norm.pdf(x_range, mu, sigma)

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_dist,
                mode='lines',
                name=f'Normal(μ={mu:.4f}, σ={sigma:.4f})',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )

        # 3. Q-Q plot
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns_clean)))
        sample_quantiles = np.sort(returns_clean)

        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(size=4, color='blue'),
                hovertemplate='Theoretical: %{x:.4f}<br>Sample: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )

        # Add diagonal reference line
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Normal Line',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=1
        )

        # 4. Price vs Volume scatter
        # Sample to avoid overplotting
        sample_size = min(10000, len(data))
        sample_indices = np.random.choice(len(data), sample_size, replace=False)
        sample_data = data.iloc[sample_indices]

        fig.add_trace(
            go.Scatter(
                x=sample_data['volume'],
                y=sample_data['price'],
                mode='markers',
                name='Price vs Volume',
                marker=dict(
                    size=3,
                    color=sample_indices,
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title='Time Index', x=1.15)
                ),
                hovertemplate='Volume: %{x:.2f}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text='Price Distribution Analysis',
                font=dict(size=16, family='Arial Black')
            ),
            height=800,
            showlegend=True
        )

        # Update axes
        fig.update_xaxes(title_text="Price (USDC)", row=1, col=1)
        fig.update_xaxes(title_text="Returns", row=1, col=2)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_xaxes(title_text="Volume", type="log", row=2, col=2)

        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Price (USDC)", row=2, col=2)

        # Statistics
        stats_dict = {
            "Price Mean": f"${data['price'].mean():.2f}",
            "Price Std": f"${data['price'].std():.2f}",
            "Returns Mean": f"{data['returns'].mean():.6f}",
            "Returns Std": f"{data['returns'].std():.6f}",
            "Skewness": f"{data['returns'].skew():.4f}",
            "Kurtosis": f"{data['returns'].kurtosis():.4f}"
        }

        return fig, stats_dict

    def plot_volatility_analysis(self, start_date=None, end_date=None, window=20):
        """Analyze price volatility patterns with interactive visualizations."""
        # Filter data
        data = self.df.copy()
        if start_date:
            start_dt = pd.to_datetime(start_date).tz_localize('UTC')
            data = data[data['timestamp'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date).tz_localize('UTC')
            data = data[data['timestamp'] <= end_dt]

        # Calculate returns and volatility
        data = data.sort_values('timestamp')
        data['returns'] = data['price'].pct_change()
        data['rolling_volatility'] = data['returns'].rolling(window=window).std() * np.sqrt(252)

        # Calculate Bollinger Bands
        data['MA'] = data['price'].rolling(window=window).mean()
        data['BB_upper'] = data['MA'] + 2 * data['price'].rolling(window=window).std()
        data['BB_lower'] = data['MA'] - 2 * data['price'].rolling(window=window).std()

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Price with Bollinger Bands',
                            f'{window}-Day Rolling Volatility',
                            'Trading Volume')
        )

        # 1. Price with Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['price'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=1),
                hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['MA'],
                mode='lines',
                name=f'MA{window}',
                line=dict(color='red', width=1, dash='dash'),
                hovertemplate='%{x}<br>MA: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['BB_upper'],
                mode='lines',
                name='Upper Band',
                line=dict(color='gray', width=1),
                showlegend=False,
                hovertemplate='%{x}<br>Upper: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['BB_lower'],
                mode='lines',
                name='Lower Band',
                line=dict(color='gray', width=1),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.2)',
                hovertemplate='%{x}<br>Lower: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # 2. Rolling volatility
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['rolling_volatility'],
                mode='lines',
                name='Volatility',
                line=dict(color='purple', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(128, 0, 128, 0.3)',
                hovertemplate='%{x}<br>Volatility: %{y:.2%}<extra></extra>'
            ),
            row=2, col=1
        )

        # Add mean volatility line
        mean_vol = data['rolling_volatility'].mean()
        fig.add_hline(
            y=mean_vol,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_vol:.2%}",
            annotation_position="right",
            row=2, col=1
        )

        # 3. Volume bars
        fig.add_trace(
            go.Bar(
                x=data['timestamp'],
                y=data['volume'],
                name='Volume',
                marker_color='green',
                opacity=0.7,
                hovertemplate='%{x}<br>Volume: %{y:,.2f}<extra></extra>'
            ),
            row=3, col=1
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text='Volatility Analysis',
                font=dict(size=16, family='Arial Black')
            ),
            height=900,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price (USDC)", row=1, col=1)
        fig.update_yaxes(title_text="Annualized Volatility", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)

        # Volatility statistics
        vol_stats = {
            "Average Volatility": f"{data['rolling_volatility'].mean():.2%}",
            "Max Volatility": f"{data['rolling_volatility'].max():.2%}",
            "Min Volatility": f"{data['rolling_volatility'].min():.2%}",
            "Current Volatility": f"{data['rolling_volatility'].iloc[-1]:.2%}"
        }

        return fig, vol_stats

    def plot_advanced_analysis(self, start_date=None, end_date=None):
        """Additional advanced analysis with interactive visualizations."""
        # Filter data
        data = self.df.copy()
        if start_date:
            start_dt = pd.to_datetime(start_date).tz_localize('UTC')
            data = data[data['timestamp'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date).tz_localize('UTC')
            data = data[data['timestamp'] <= end_dt]

        # Resample to hourly for cleaner visualization
        hourly = data.set_index('timestamp')['price'].resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        hourly['volume'] = data.set_index('timestamp')['volume'].resample('1H').sum()
        hourly['trade_count'] = data.set_index('timestamp')['price'].resample('1H').count()
        hourly['vwap'] = (data.set_index('timestamp')['price'] * data.set_index('timestamp')['volume']).resample(
            '1H').sum() / hourly['volume']

        # Calculate additional metrics
        hourly['spread'] = (hourly['high'] - hourly['low']) / hourly['close'] * 100
        hourly['returns'] = hourly['close'].pct_change()

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Spread %', 'Trade Intensity',
                            'VWAP vs Close Price', 'Volume-Weighted Returns'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )

        # 1. Price spread
        fig.add_trace(
            go.Scatter(
                x=hourly.index,
                y=hourly['spread'],
                mode='lines',
                name='Price Spread %',
                line=dict(color='orange', width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 165, 0, 0.3)',
                hovertemplate='%{x}<br>Spread: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # 2. Trade intensity
        fig.add_trace(
            go.Scatter(
                x=hourly.index,
                y=hourly['trade_count'],
                mode='lines',
                name='Trades per Hour',
                line=dict(color='green', width=1),
                hovertemplate='%{x}<br>Trades: %{y}<extra></extra>'
            ),
            row=1, col=2
        )

        # 3. VWAP vs Close
        fig.add_trace(
            go.Scatter(
                x=hourly.index,
                y=hourly['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=1),
                hovertemplate='%{x}<br>Close: $%{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=hourly.index,
                y=hourly['vwap'],
                mode='lines',
                name='VWAP',
                line=dict(color='red', width=1, dash='dash'),
                hovertemplate='%{x}<br>VWAP: $%{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )

        # Add volume on secondary y-axis
        fig.add_trace(
            go.Bar(
                x=hourly.index,
                y=hourly['volume'],
                name='Volume',
                marker_color='lightgray',
                opacity=0.3,
                yaxis='y2',
                hovertemplate='%{x}<br>Volume: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1,
            secondary_y=True
        )

        # 4. Volume-weighted returns scatter
        fig.add_trace(
            go.Scatter(
                x=hourly['volume'],
                y=hourly['returns'] * 100,
                mode='markers',
                name='Returns vs Volume',
                marker=dict(
                    size=5,
                    color=hourly['spread'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Spread %', x=1.15)
                ),
                hovertemplate='Volume: %{x:,.0f}<br>Return: %{y:.2f}%<br>Spread: %{marker.color:.2f}%<extra></extra>'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text='Advanced Market Analysis',
                font=dict(size=16, family='Arial Black')
            ),
            height=800,
            showlegend=True
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Volume", type="log", row=2, col=2)

        fig.update_yaxes(title_text="Spread %", row=1, col=1)
        fig.update_yaxes(title_text="Trades per Hour", row=1, col=2)
        fig.update_yaxes(title_text="Price (USDC)", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Returns %", row=2, col=2)

        return fig


# Main Streamlit App
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



    st.title("📊 ETH/USDC Price Dynamics Analyzer")
    st.markdown("---")

    # Initialize session state
    if 'viz' not in st.session_state:
        try:
            st.session_state.viz = PriceDynamicsVisualizer('Kraken-ETHUSDC-2024-2025.parquet')
            st.success(f"✅ Loaded {len(st.session_state.viz.df):,} records")
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.stop()

    viz = st.session_state.viz

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


    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    # Date range selection
    st.sidebar.subheader("📅 Date Range Selection")

    min_date = viz.df['timestamp'].min().date()
    max_date = viz.df['timestamp'].max().date()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=30),
            min_value=min_date,
            max_value=max_date
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

    # Data info
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Data Info")
    st.sidebar.info(f"""
    **Date Range**: {min_date} to {max_date}
    **Total Records**: {len(viz.df):,}
    """)

    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Candlestick Chart",
        "🗓️ Price Heatmap",
        "📊 Distribution Analysis",
        "📉 Volatility Analysis",
        "🔬 Advanced Analysis"
    ])

    with tab1:
        st.subheader("📈 Candlestick Chart with Volume")

        # Resample frequency selection
        resample_freq = st.select_slider(
            "Select Time Interval",
            options=['5T', '15T', '30T', '1H', '4H', '1D'],
            value='1H',
            help="Choose the time interval for candlestick aggregation"
        )

        # Generate candlestick chart
        with st.spinner("Generating candlestick chart..."):
            fig = viz.plot_candlestick_chart(
                start_date=start_date,
                end_date=end_date,
                resample=resample_freq
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🗓️ Price and Volume Patterns by Hour and Day")

        with st.spinner("Generating heatmap..."):
            fig = viz.plot_price_heatmap(
                start_date=start_date,
                end_date=end_date
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info("💡 **Tip**: Hover over cells to see exact values. Darker colors indicate higher values.")

    with tab3:
        st.subheader("📊 Price and Returns Distribution Analysis")

        with st.spinner("Analyzing distributions..."):
            fig, stats_dict = viz.plot_price_distribution(
                start_date=start_date,
                end_date=end_date
            )
            st.plotly_chart(fig, use_container_width=True)

        # Display statistics
        st.subheader("📈 Distribution Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Price Mean", stats_dict["Price Mean"])
            st.metric("Price Std", stats_dict["Price Std"])

        with col2:
            st.metric("Returns Mean", stats_dict["Returns Mean"])
            st.metric("Returns Std", stats_dict["Returns Std"])

        with col3:
            st.metric("Skewness", stats_dict["Skewness"])
            st.metric("Kurtosis", stats_dict["Kurtosis"])

    with tab4:
        st.subheader("📉 Volatility Analysis with Bollinger Bands")

        # Volatility window selection
        vol_window = st.slider(
            "Select Volatility Window (days)",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )

        with st.spinner("Analyzing volatility..."):
            fig, vol_stats = viz.plot_volatility_analysis(
                start_date=start_date,
                end_date=end_date,
                window=vol_window
            )
            st.plotly_chart(fig, use_container_width=True)

        # Display volatility statistics
        st.subheader("📊 Volatility Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Average", vol_stats["Average Volatility"])
        with col2:
            st.metric("Maximum", vol_stats["Max Volatility"])
        with col3:
            st.metric("Minimum", vol_stats["Min Volatility"])
        with col4:
            st.metric("Current", vol_stats["Current Volatility"])

    with tab5:
        st.subheader("🔬 Advanced Market Analysis")

        with st.spinner("Performing advanced analysis..."):
            fig = viz.plot_advanced_analysis(
                start_date=start_date,
                end_date=end_date
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info("""
        💡 **Analysis Guide**:
        - **Price Spread %**: Shows the intraday volatility (high-low range)
        - **Trade Intensity**: Number of trades per hour
        - **VWAP**: Volume-weighted average price vs closing price
        - **Volume-Weighted Returns**: Relationship between returns and volume
        """)



if __name__ == "__main__":
    main()
