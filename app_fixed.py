import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
import calendar
import pytz
import io
import zipfile
import warnings
from scipy import stats
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Futures Seasonality Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #FAFAFA;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #262730;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #434651;
        margin: 0.5rem 0;
    }
    .pattern-highlight {
        background: rgba(0, 212, 170, 0.1);
        border-left: 3px solid #00D4AA;
        padding: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background: rgba(242, 54, 69, 0.1);
        border-left: 3px solid #F23645;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {}

class DataLoader:
    """Handles loading and preprocessing of various data formats"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls', 'parquet', 'txt']
        self.date_columns = ['date', 'datetime', 'timestamp', 'time', 'Date', 'DateTime', 'Timestamp', 'Time']
        self.price_columns = {
            'open': ['open', 'Open', 'OPEN', 'o', 'O'],
            'high': ['high', 'High', 'HIGH', 'h', 'H'],
            'low': ['low', 'Low', 'LOW', 'l', 'L'],
            'close': ['close', 'Close', 'CLOSE', 'c', 'C'],
            'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'Vol', 'VOL', 'v', 'V']
        }
    
    def load_file(self, uploaded_file):
        """Load data from uploaded file"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ['csv', 'txt']:
                df = self._load_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = self._load_excel(uploaded_file)
            elif file_extension == 'parquet':
                df = self._load_parquet(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
            
            if df is None:
                return None
            
            # Process the data
            df = self._process_data(df)
            return df
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    def _load_csv(self, uploaded_file):
        """Load CSV file with automatic delimiter detection"""
        try:
            # Read a sample to detect delimiter
            sample = uploaded_file.read(1024).decode('utf-8')
            uploaded_file.seek(0)
            
            # Count delimiters
            delimiters = [',', ';', '\t', '|']
            delimiter_counts = {d: sample.count(d) for d in delimiters}
            best_delimiter = max(delimiter_counts.keys(), key=lambda x: delimiter_counts[x])
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, delimiter=best_delimiter, encoding=encoding)
                    if len(df.columns) > 1:  # Successfully parsed
                        return df
                except:
                    continue
            
            # Fallback to default CSV reading
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)
            
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            return None
    
    def _load_excel(self, uploaded_file):
        """Load Excel file"""
        try:
            df = pd.read_excel(uploaded_file, sheet_name=0)
            return df
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            return None
    
    def _load_parquet(self, uploaded_file):
        """Load Parquet file"""
        try:
            df = pd.read_parquet(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error reading Parquet file: {str(e)}")
            return None
    
    def _process_data(self, df):
        """Process and clean the loaded data"""
        if df is None or df.empty:
            return None
        
        # Identify date column
        date_col = self._identify_date_column(df)
        if date_col is None:
            st.error("Could not identify date/time column.")
            return None
        
        # Convert to datetime and set as index
        try:
            df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
            df = df.set_index(date_col)
            df.index.name = 'datetime'
        except Exception as e:
            st.error(f"Error converting date column: {str(e)}")
            return None
        
        # Identify and standardize price columns
        price_mapping = self._identify_price_columns(df)
        if not price_mapping:
            st.error("Could not identify OHLC price columns.")
            return None
        
        # Rename columns to standard names
        df = df.rename(columns=price_mapping)
        
        # Keep only OHLC columns (and volume if available)
        required_cols = ['Open', 'High', 'Low', 'Close']
        optional_cols = ['Volume']
        
        available_cols = [col for col in required_cols if col in df.columns]
        available_cols.extend([col for col in optional_cols if col in df.columns])
        
        df = df[available_cols]
        
        # Clean data
        df = df.dropna(subset=required_cols)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Ensure positive prices
        for col in required_cols:
            df = df[df[col] > 0]
        
        # Sort by date
        df = df.sort_index()
        
        if len(df) == 0:
            st.error("No valid data remaining after cleaning.")
            return None
        
        return df
    
    def _identify_date_column(self, df):
        """Identify the date/time column"""
        # First, check for common date column names
        for col in df.columns:
            if col.lower() in [d.lower() for d in self.date_columns]:
                return col
        
        # Then, try to find columns that can be converted to datetime
        for col in df.columns:
            try:
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    pd.to_datetime(sample, infer_datetime_format=True)
                    return col
            except:
                continue
        
        return None
    
    def _identify_price_columns(self, df):
        """Identify and map price columns to standard names"""
        mapping = {}
        
        for standard_name, possible_names in self.price_columns.items():
            for col in df.columns:
                if col.lower() in [name.lower() for name in possible_names]:
                    if standard_name == 'volume':
                        mapping[col] = 'Volume'
                    else:
                        mapping[col] = standard_name.capitalize()
                    break
        
        # Check if we have the required OHLC columns
        required = {'Open', 'High', 'Low', 'Close'}
        if not required.issubset(set(mapping.values())):
            return None
        
        return mapping

class SeasonalityAnalyzer:
    """Analyzes various types of seasonality patterns"""
    
    def __init__(self):
        self.min_observations = 10
    
    def detect_timeframe(self, data):
        """Detect data timeframe based on index frequency"""
        try:
            unique_hours = data.index.hour.nunique()
            unique_minutes = data.index.minute.nunique()
            
            if unique_hours <= 1 and unique_minutes <= 1:
                return "daily"
            elif unique_hours > 10:
                return "intraday_minutes"
            elif unique_hours > 1:
                return "intraday_hours"
            else:
                return "mixed"
        except:
            return "unknown"
    
    def analyze_intraday_patterns(self, data, trading_hours=None):
        """Analyze intraday seasonality patterns"""
        try:
            timeframe = self.detect_timeframe(data)
            
            if timeframe == "daily":
                return None  # Not suitable for intraday analysis
            
            data_copy = data.copy()
            data_copy['returns'] = data_copy['Close'].pct_change() * 100
            data_copy = data_copy.dropna()
            
            if len(data_copy) == 0:
                return None
            
            # Extract time features
            data_copy['hour'] = data_copy.index.hour
            data_copy['minute'] = data_copy.index.minute
            
            # Check if data has meaningful time variation
            unique_hours = data_copy['hour'].nunique()
            unique_minutes = data_copy['minute'].nunique()
            
            if unique_hours <= 1 and unique_minutes <= 1:
                return None
            
            # Filter by trading hours if specified
            if trading_hours and (unique_hours > 1 or unique_minutes > 1):
                start_time, end_time = trading_hours
                start_hour = start_time.hour + start_time.minute / 60
                end_hour = end_time.hour + end_time.minute / 60
                
                data_copy['hour_decimal'] = data_copy['hour'] + data_copy['minute'] / 60
                data_copy = data_copy[
                    (data_copy['hour_decimal'] >= start_hour) & 
                    (data_copy['hour_decimal'] <= end_hour)
                ]
            
            # Adjust minimum observations based on data frequency
            if unique_hours > 10:  # Likely minute-level data
                min_obs = self.min_observations
            elif unique_hours > 5:  # Likely hourly data
                min_obs = 5
            else:  # Few unique hours
                min_obs = 3
            
            if len(data_copy) < min_obs:
                return None
            
            result = data_copy[['hour', 'minute', 'returns', 'Close']].copy()
            result['datetime'] = data_copy.index
            
            return result
            
        except Exception as e:
            st.error(f"Error in intraday analysis: {str(e)}")
            return None
    
    def analyze_weekly_patterns(self, data):
        """Analyze day-of-week seasonality patterns"""
        try:
            data_copy = data.copy()
            data_copy['returns'] = data_copy['Close'].pct_change() * 100
            data_copy = data_copy.dropna()
            
            if len(data_copy) == 0:
                return None
            
            # Extract day of week (0=Monday, 6=Sunday)
            data_copy['day_of_week'] = data_copy.index.dayofweek
            data_copy['year'] = data_copy.index.year
            data_copy['week'] = data_copy.index.isocalendar().week
            
            # Filter to only trading days (Monday-Friday)
            data_copy = data_copy[data_copy['day_of_week'] < 5]
            
            if len(data_copy) < self.min_observations:
                return None
            
            result = data_copy[['day_of_week', 'year', 'week', 'returns', 'Close']].copy()
            result['datetime'] = data_copy.index
            
            return result
            
        except Exception as e:
            st.error(f"Error in weekly analysis: {str(e)}")
            return None
    
    def analyze_monthly_patterns(self, data):
        """Analyze day-of-month seasonality patterns"""
        try:
            data_copy = data.copy()
            data_copy['returns'] = data_copy['Close'].pct_change() * 100
            data_copy = data_copy.dropna()
            
            if len(data_copy) == 0:
                return None
            
            # Extract day of month
            data_copy['day_of_month'] = data_copy.index.day
            data_copy['month'] = data_copy.index.month
            data_copy['year'] = data_copy.index.year
            
            if len(data_copy) < self.min_observations:
                return None
            
            result = data_copy[['day_of_month', 'month', 'year', 'returns', 'Close']].copy()
            result['datetime'] = data_copy.index
            
            return result
            
        except Exception as e:
            st.error(f"Error in monthly analysis: {str(e)}")
            return None
    
    def analyze_seasonal_patterns(self, data):
        """Analyze seasonal patterns using % gain relative to the first trading day of each year (trading-day index)"""
        try:
            if data is None or len(data) == 0:
                return None

            data_sorted = data.sort_index()
            yearly_data = []
            max_trading_days = 0

            for year, year_df in data_sorted.groupby(data_sorted.index.year):
                year_df = year_df.sort_index()
                if len(year_df) < 2:
                    continue
                # Use the first available close of the year as baseline
                first_price = year_df['Close'].iloc[0]
                year_df = year_df.copy()
                year_df['month'] = year_df.index.month
                year_df['day'] = year_df.index.day
                year_df['day_of_year'] = year_df.index.dayofyear
                year_df['pct_gain_from_jan1'] = (year_df['Close'] / first_price - 1) * 100
                year_df['year'] = year
                # trading day index within the year (1,2,3,...)
                year_df['trading_day'] = np.arange(1, len(year_df) + 1)
                yearly_data.append(year_df)
                max_trading_days = max(max_trading_days, len(year_df))

            if not yearly_data:
                return None

            result = pd.concat(yearly_data)
            seasonal_avg = result.groupby('trading_day')['pct_gain_from_jan1'].agg(['mean', 'std', 'count']).reset_index()
            seasonal_avg.columns = ['trading_day', 'avg_gain', 'std_gain', 'count']

            if len(seasonal_avg) < self.min_observations:
                return None

            return {
                'data': result,
                'seasonal_avg': seasonal_avg,
                'years': result['year'].unique(),
                'max_trading_days': max_trading_days
            }

        except Exception as e:
            st.error(f"Error in seasonal analysis: {str(e)}")
            return None

    def analyze_seasonal_patterns_calendar(self, data):
        """Analyze seasonal patterns aligned by calendar day-of-year (1..366), baseline = first trading day's close."""
        try:
            if data is None or len(data) == 0:
                return None

            data_sorted = data.sort_index()
            yearly_data = []

            for year, year_df in data_sorted.groupby(data_sorted.index.year):
                year_df = year_df.sort_index()
                if len(year_df) < 2:
                    continue
                first_price = year_df['Close'].iloc[0]
                year_df = year_df.copy()
                year_df['year'] = year
                year_df['day_of_year'] = year_df.index.dayofyear
                year_df['month'] = year_df.index.month
                year_df['day'] = year_df.index.day
                year_df['pct_gain_from_jan1'] = (year_df['Close'] / first_price - 1) * 100
                yearly_data.append(year_df)

            if not yearly_data:
                return None

            result = pd.concat(yearly_data)
            seasonal_avg = result.groupby('day_of_year')['pct_gain_from_jan1'].agg(['mean', 'std', 'count']).reset_index()
            seasonal_avg.columns = ['day_of_year', 'avg_gain', 'std_gain', 'count']

            if len(seasonal_avg) < self.min_observations:
                return None

            return {
                'data': result,
                'seasonal_avg': seasonal_avg,
                'years': result['year'].unique()
            }
        except Exception as e:
            st.error(f"Error in calendar seasonal analysis: {str(e)}")
            return None

class Visualizer:
    """Create interactive visualizations"""
    
    def __init__(self):
        self.colors = {
            'primary': '#00D4AA',
            'secondary': '#F23645',
            'background': '#0E1117',
            'surface': '#262730',
            'text': '#FAFAFA'
        }
    
    def create_price_chart(self, data):
        """Create basic price chart"""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=self.colors['primary'], width=2)
            ))
            
            fig.update_layout(
                title="Price History",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_dark",
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background'],
                font=dict(color=self.colors['text']),
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating price chart: {str(e)}")
            return go.Figure()
    
    def create_hourly_returns_chart(self, hourly_patterns):
        """Create average returns by hour chart"""
        try:
            if hourly_patterns is None or len(hourly_patterns) == 0:
                return go.Figure()
            
            avg_returns = hourly_patterns.groupby('hour')['returns'].mean()
            std_returns = hourly_patterns.groupby('hour')['returns'].std()
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=avg_returns.index,
                y=avg_returns.values,
                name='Average Returns',
                marker_color=[self.colors['primary'] if x >= 0 else self.colors['secondary'] for x in avg_returns.values],
                error_y=dict(type='data', array=std_returns.values, visible=True)
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            
            fig.update_layout(
                title="Average Returns by Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Average Returns (%)",
                template="plotly_dark",
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background'],
                font=dict(color=self.colors['text']),
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating hourly returns chart: {str(e)}")
            return go.Figure()
    
    def create_weekly_pattern_chart(self, weekly_patterns):
        """Create day-of-week pattern chart"""
        try:
            if weekly_patterns is None or len(weekly_patterns) == 0:
                return go.Figure()
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            avg_returns = weekly_patterns.groupby('day_of_week')['returns'].mean()
            std_returns = weekly_patterns.groupby('day_of_week')['returns'].std()
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=[day_names[i] for i in avg_returns.index],
                y=avg_returns.values,
                name='Average Returns',
                marker_color=[self.colors['primary'] if x >= 0 else self.colors['secondary'] for x in avg_returns.values],
                error_y=dict(type='data', array=std_returns.values, visible=True)
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            
            fig.update_layout(
                title="Average Returns by Day of Week",
                xaxis_title="Day of Week",
                yaxis_title="Average Returns (%)",
                template="plotly_dark",
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background'],
                font=dict(color=self.colors['text']),
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating weekly pattern chart: {str(e)}")
            return go.Figure()
    
    def create_monthly_pattern_chart(self, monthly_patterns):
        """Create day-of-month pattern chart"""
        try:
            if monthly_patterns is None or len(monthly_patterns) == 0:
                return go.Figure()
            
            avg_returns = monthly_patterns.groupby('day_of_month')['returns'].mean()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=avg_returns.index,
                y=avg_returns.values,
                mode='lines+markers',
                name='Average Returns',
                line=dict(color=self.colors['primary'], width=2),
                marker=dict(size=6)
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            
            fig.update_layout(
                title="Average Returns by Day of Month",
                xaxis_title="Day of Month",
                yaxis_title="Average Returns (%)",
                template="plotly_dark",
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background'],
                font=dict(color=self.colors['text']),
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating monthly pattern chart: {str(e)}")
            return go.Figure()
    
    def create_seasonal_overlay_chart(self, seasonal_results):
        """Create seasonal overlay chart showing % gain from first trading day of each year (trading-day aligned)"""
        try:
            if seasonal_results is None:
                return go.Figure()
            
            data = seasonal_results['data']
            seasonal_avg = seasonal_results['seasonal_avg']
            years = seasonal_results['years']
            max_days = seasonal_results.get('max_trading_days', seasonal_avg['trading_day'].max())
            
            fig = go.Figure()
            
            # Add individual year lines (lighter colors)
            colors_cycle = ['rgba(100, 100, 255, 0.3)', 'rgba(255, 100, 100, 0.3)', 'rgba(100, 255, 100, 0.3)', 
                           'rgba(255, 255, 100, 0.3)', 'rgba(255, 100, 255, 0.3)', 'rgba(100, 255, 255, 0.3)']
            
            for i, year in enumerate(sorted(years)):
                year_data = data[data['year'] == year].sort_values('trading_day')
                fig.add_trace(go.Scatter(
                    x=year_data['trading_day'],
                    y=year_data['pct_gain_from_jan1'],
                    mode='lines',
                    name=str(year),
                    line=dict(color=colors_cycle[i % len(colors_cycle)], width=1),
                    opacity=0.6,
                    hoverinfo='x+y+name'
                ))
            
            # Add average line (bold)
            fig.add_trace(go.Scatter(
                x=seasonal_avg['trading_day'],
                y=seasonal_avg['avg_gain'],
                mode='lines',
                name='Average',
                line=dict(color=self.colors['primary'], width=3),
                opacity=1.0
            ))
            
            # Add confidence bands
            upper_band = seasonal_avg['avg_gain'] + seasonal_avg['std_gain']
            lower_band = seasonal_avg['avg_gain'] - seasonal_avg['std_gain']
            
            fig.add_trace(go.Scatter(
                x=list(seasonal_avg['trading_day']) + list(seasonal_avg['trading_day'][::-1]),
                y=list(upper_band) + list(lower_band[::-1]),
                fill='toself',
                fillcolor='rgba(0, 212, 170, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Std Dev',
                showlegend=True
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            
            fig.update_layout(
                title="Seasonal Pattern: % Gain from First Trading Day of Year",
                xaxis_title="Trading Day Index (1 = first trading day of year)",
                yaxis_title="% Gain from First Trading Day",
                template="plotly_dark",
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background'],
                font=dict(color=self.colors['text']),
                height=600,
                xaxis=dict(range=[1, max_days])
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating seasonal overlay chart: {str(e)}")
            return go.Figure()

    def create_calendar_overlay_chart(self, seasonal_results, day_tick_stride=7):
        """Create calendar day-of-year aligned overlay chart (x-axis = day_of_year 1..366)."""
        try:
            if seasonal_results is None:
                return go.Figure()

            data = seasonal_results['data']
            seasonal_avg = seasonal_results['seasonal_avg']
            years = seasonal_results['years']

            fig = go.Figure()

            colors_cycle = ['rgba(100, 100, 255, 0.3)', 'rgba(255, 100, 100, 0.3)', 'rgba(100, 255, 100, 0.3)', 
                           'rgba(255, 255, 100, 0.3)', 'rgba(255, 100, 255, 0.3)', 'rgba(100, 255, 255, 0.3)']

            for i, year in enumerate(sorted(years)):
                year_data = data[data['year'] == year].sort_values('day_of_year')
                fig.add_trace(go.Scatter(
                    x=year_data['day_of_year'],
                    y=year_data['pct_gain_from_jan1'],
                    mode='lines',
                    name=str(year),
                    line=dict(color=colors_cycle[i % len(colors_cycle)], width=1),
                    opacity=0.6,
                    hoverinfo='x+y+name'
                ))

            fig.add_trace(go.Scatter(
                x=seasonal_avg['day_of_year'],
                y=seasonal_avg['avg_gain'],
                mode='lines',
                name='Average',
                line=dict(color=self.colors['primary'], width=3),
                opacity=1.0
            ))

            upper_band = seasonal_avg['avg_gain'] + seasonal_avg['std_gain']
            lower_band = seasonal_avg['avg_gain'] - seasonal_avg['std_gain']

            fig.add_trace(go.Scatter(
                x=list(seasonal_avg['day_of_year']) + list(seasonal_avg['day_of_year'][::-1]),
                y=list(upper_band) + list(lower_band[::-1]),
                fill='toself',
                fillcolor='rgba(0, 212, 170, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Std Dev',
                showlegend=True
            ))

            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)

            # Month boundary ticks (approx cumulative day-of-year at month starts in non-leap year)
            month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            fig.update_layout(
                title="Seasonal Pattern (Calendar): % Gain from First Trading Day of Year",
                xaxis_title="Day of Year",
                yaxis_title="% Gain from First Trading Day",
                template="plotly_dark",
                paper_bgcolor=self.colors['background'],
                plot_bgcolor=self.colors['background'],
                font=dict(color=self.colors['text']),
                height=600,
                xaxis=dict(
                    tickmode='array',
                    tickvals=month_starts[::max(1, int(len(month_starts)/max(1, int(12/day_tick_stride))))],
                    ticktext=month_labels[::max(1, int(len(month_labels)/max(1, int(12/day_tick_stride))))]
                )
            )

            return fig
        except Exception as e:
            st.error(f"Error creating calendar overlay chart: {str(e)}")
            return go.Figure()

class StatisticalAnalyzer:
    """Perform statistical analysis on patterns"""
    
    def __init__(self):
        self.significance_level = 0.05
    
    def calculate_basic_stats(self, data):
        """Calculate basic statistical measures"""
        try:
            if data is None or data.empty:
                return pd.DataFrame()
            
            returns = data['Close'].pct_change().dropna() * 100
            
            stats_dict = {
                'Metric': [
                    'Mean Return (%)',
                    'Std Deviation (%)',
                    'Skewness',
                    'Kurtosis',
                    'Sharpe Ratio',
                    'Min Return (%)',
                    'Max Return (%)',
                    'VaR (95%)',
                    'CVaR (95%)',
                    'Positive Days (%)'
                ],
                'Value': [
                    f"{returns.mean():.4f}",
                    f"{returns.std():.4f}",
                    f"{returns.skew():.4f}",
                    f"{returns.kurtosis():.4f}",
                    f"{returns.mean() / returns.std():.4f}" if returns.std() > 0 else "0",
                    f"{returns.min():.4f}",
                    f"{returns.max():.4f}",
                    f"{np.percentile(returns, 5):.4f}",
                    f"{returns[returns <= np.percentile(returns, 5)].mean():.4f}",
                    f"{(returns > 0).mean() * 100:.2f}"
                ]
            }
            
            return pd.DataFrame(stats_dict)
            
        except Exception as e:
            st.error(f"Error calculating basic stats: {str(e)}")
            return pd.DataFrame()
    
    def test_pattern_significance(self, patterns, group_column):
        """Test statistical significance of patterns using ANOVA"""
        try:
            if patterns is None or len(patterns) == 0:
                return None
            
            # Group data by pattern
            groups = [group['returns'].values for name, group in patterns.groupby(group_column)]
            
            # Remove empty groups
            groups = [group for group in groups if len(group) > 0]
            
            if len(groups) < 2:
                return None
            
            # Perform one-way ANOVA
            f_stat, p_value = f_oneway(*groups)
            
            # Effect size (eta squared)
            ss_between = sum(len(group) * (np.mean(group) - patterns['returns'].mean())**2 for group in groups)
            ss_total = sum((patterns['returns'] - patterns['returns'].mean())**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            return {
                'f_statistic': f_stat,
                'p_value': p_value,
                'is_significant': p_value < self.significance_level,
                'effect_size': eta_squared
            }
            
        except Exception as e:
            st.error(f"Error testing pattern significance: {str(e)}")
            return None

def main():
    st.title("🔮 Futures Seasonality Analysis Tool")
    st.markdown("### Focus on Day-of-Year Seasonality with % Gain relative to Jan 1")
    
    # Initialize components
    data_loader = DataLoader()
    seasonality_analyzer = SeasonalityAnalyzer()
    visualizer = Visualizer()
    statistical_analyzer = StatisticalAnalyzer()
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your futures data",
            type=['csv', 'xlsx', 'xls', 'parquet', 'txt'],
            help="Supported formats: CSV, Excel, Parquet, TXT. File should contain OHLC data with timestamps."
        )
        
        if uploaded_file is not None:
            # Load data
            with st.spinner("Loading and processing data..."):
                data = data_loader.load_file(uploaded_file)
                if data is not None:
                    st.session_state.data = data
                    st.success(f"✅ Loaded {len(data)} records")
                    
                    # Show data info
                    st.subheader("📊 Data Summary")
                    st.write(f"**Date Range:** {data.index.min().date()} to {data.index.max().date()}")
                    st.write(f"**Records:** {len(data):,}")
                    st.write(f"**Columns:** {', '.join(data.columns)}")
                    
                    # Detect timeframe
                    timeframe = seasonality_analyzer.detect_timeframe(data)
                    st.write(f"**Detected Timeframe:** {timeframe.replace('_', ' ').title()}")
        
        # Configuration options
        if st.session_state.data is not None:
            st.header("⚙️ Analysis Settings")
            
            # Trading hours configuration
            st.subheader("Trading Hours (for intraday)")
            trading_start = st.time_input("Market Open", value=datetime.strptime("09:30", "%H:%M").time())
            trading_end = st.time_input("Market Close", value=datetime.strptime("16:00", "%H:%M").time())
            
            # Analysis parameters
            st.subheader("Analysis Parameters")
            min_observations = st.slider("Minimum observations per pattern", 5, 50, 10)
            confidence_level = st.slider("Confidence level (%)", 90, 99, 95) / 100
            
            # Store settings in session state
            st.session_state.trading_hours = (trading_start, trading_end)
            st.session_state.min_observations = min_observations
            st.session_state.confidence_level = confidence_level
    
    # Main content area
    if st.session_state.data is None:
        st.info("👆 Please upload a data file to begin analysis")
        
        # Show sample data format
        st.subheader("📋 Expected Data Format")
        st.markdown("""
        Your file should contain the following columns:
        - **Date/Time column** (various formats supported)
        - **Open** - Opening price
        - **High** - High price  
        - **Low** - Low price
        - **Close** - Closing price
        - **Volume** (optional)
        
        The tool will automatically detect column names and date formats.
        """)
        
        return
    
    # Data loaded - show analysis options
    data = st.session_state.data
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Data Overview", 
        "⏰ Intraday Patterns", 
        "📅 Weekly Patterns", 
        "📆 Monthly Patterns", 
        "🌟 Seasonal Patterns"
    ])
    
    with tab1:
        show_data_overview(data, visualizer, statistical_analyzer)
    
    with tab2:
        show_intraday_analysis(data, seasonality_analyzer, visualizer, statistical_analyzer)
    
    with tab3:
        show_weekly_analysis(data, seasonality_analyzer, visualizer, statistical_analyzer)
    
    with tab4:
        show_monthly_analysis(data, seasonality_analyzer, visualizer, statistical_analyzer)
    
    with tab5:
        show_seasonal_analysis(data, seasonality_analyzer, visualizer, statistical_analyzer)

def show_data_overview(data, visualizer, statistical_analyzer):
    """Display data overview and basic statistics"""
    st.header("📈 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        st.metric("Date Range", f"{(data.index.max() - data.index.min()).days} days")
    with col3:
        current_price = data['Close'].iloc[-1] if len(data) > 0 else 0
        st.metric("Latest Price", f"${current_price:.2f}")
    with col4:
        price_change = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100 if len(data) > 1 else 0
        st.metric("Total Return", f"{price_change:.2f}%")
    
    # Price chart
    st.subheader("Price History")
    fig = visualizer.create_price_chart(data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Basic statistics
    st.subheader("Statistical Summary")
    stats = statistical_analyzer.calculate_basic_stats(data)
    
    if not stats.empty:
        st.dataframe(stats, use_container_width=True)

def show_intraday_analysis(data, seasonality_analyzer, visualizer, statistical_analyzer):
    """Display intraday seasonality analysis with improved detection"""
    st.header("⏰ Intraday Seasonality Patterns")
    
    if len(data) == 0:
        st.warning("No data available for analysis")
        return
    
    # Detect data timeframe
    timeframe = seasonality_analyzer.detect_timeframe(data)
    unique_hours = data.index.hour.nunique()
    unique_minutes = data.index.minute.nunique()
    
    if timeframe == "daily":
        st.info("📅 **Daily or higher timeframe data detected.** Intraday analysis requires minute/hourly data.")
        st.markdown("""
        **Data Timeframe:** Daily or higher  
        **Recommendation:** Use Weekly, Monthly, or Seasonal Patterns tabs for this data.
        """)
        return
    
    # Show timeframe info
    st.info(f"📊 **Intraday data detected:** {unique_hours} unique hours, {unique_minutes} unique minutes")
    
    # Get trading hours from session state
    trading_hours = getattr(st.session_state, 'trading_hours', (datetime.strptime("09:30", "%H:%M").time(), datetime.strptime("16:00", "%H:%M").time()))
    
    with st.spinner("Analyzing intraday patterns..."):
        # Calculate hourly patterns
        hourly_patterns = seasonality_analyzer.analyze_intraday_patterns(data, trading_hours)
        
        if hourly_patterns is not None and len(hourly_patterns) > 0:
            # Statistical significance
            significance = statistical_analyzer.test_pattern_significance(hourly_patterns, 'hour')
            
            col1, col2 = st.columns(2)
            
            with col2:
                # Average returns by hour
                fig = visualizer.create_hourly_returns_chart(hourly_patterns)
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistical results
            st.subheader("📊 Statistical Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_hour = hourly_patterns.groupby('hour')['returns'].mean().idxmax()
                avg_return = hourly_patterns.groupby('hour')['returns'].mean().loc[best_hour]
                st.metric("Best Hour", f"{best_hour}:00", f"{avg_return:.4f}%")
            
            with col2:
                worst_hour = hourly_patterns.groupby('hour')['returns'].mean().idxmin()
                avg_return = hourly_patterns.groupby('hour')['returns'].mean().loc[worst_hour]
                st.metric("Worst Hour", f"{worst_hour}:00", f"{avg_return:.4f}%")
            
            with col3:
                p_value = significance.get('p_value', 1.0) if significance else 1.0
                is_significant = p_value < (1 - getattr(st.session_state, 'confidence_level', 0.95))
                st.metric("Statistically Significant", "Yes" if is_significant else "No", f"p={p_value:.4f}")
            
            # Detailed statistics table
            if st.checkbox("Show Detailed Hourly Statistics"):
                hourly_stats = hourly_patterns.groupby('hour')['returns'].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(4)
                hourly_stats.columns = ['Observations', 'Avg Return (%)', 'Std Dev (%)', 'Min (%)', 'Max (%)']
                st.dataframe(hourly_stats)
        else:
            st.warning(f"""Not enough intraday data for pattern analysis.  
            **Data points after filtering:** {len(hourly_patterns) if hourly_patterns is not None else 0}  
            **Trading hours filter:** {trading_hours[0]} - {trading_hours[1]}  
            **Try:** Adjust trading hours in the sidebar or use data with more frequent timestamps.""")

def show_weekly_analysis(data, seasonality_analyzer, visualizer, statistical_analyzer):
    """Display day-of-week seasonality analysis"""
    st.header("📅 Day-of-Week Seasonality")
    
    with st.spinner("Analyzing weekly patterns..."):
        weekly_patterns = seasonality_analyzer.analyze_weekly_patterns(data)
        
        if weekly_patterns is not None and len(weekly_patterns) > 0:
            # Statistical significance
            significance = statistical_analyzer.test_pattern_significance(weekly_patterns, 'day_of_week')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Day of week returns
                fig = visualizer.create_weekly_pattern_chart(weekly_patterns)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Show statistics
                daily_stats = weekly_patterns.groupby('day_of_week')['returns'].agg([
                    'count', 'mean', 'std'
                ]).round(4)
                
                # Add day names
                day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
                daily_stats.index = [day_names.get(idx, f'Day {idx}') for idx in daily_stats.index]
                daily_stats.columns = ['Observations', 'Avg Return (%)', 'Std Dev (%)']
                
                st.dataframe(daily_stats)
            
            # Best/worst days
            best_day_idx = weekly_patterns.groupby('day_of_week')['returns'].mean().idxmax()
            worst_day_idx = weekly_patterns.groupby('day_of_week')['returns'].mean().idxmin()
            
            day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
            best_day = day_names.get(best_day_idx, f'Day {best_day_idx}')
            worst_day = day_names.get(worst_day_idx, f'Day {worst_day_idx}')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Day", best_day)
            with col2:
                st.metric("Worst Day", worst_day)
            with col3:
                p_value = significance.get('p_value', 1.0) if significance else 1.0
                is_significant = p_value < (1 - getattr(st.session_state, 'confidence_level', 0.95))
                st.metric("Significant Pattern", "Yes" if is_significant else "No")
        else:
            st.warning("Not enough data for weekly pattern analysis.")

def show_monthly_analysis(data, seasonality_analyzer, visualizer, statistical_analyzer):
    """Display day-of-month seasonality analysis"""
    st.header("📆 Day-of-Month Seasonality")
    
    with st.spinner("Analyzing monthly patterns..."):
        monthly_patterns = seasonality_analyzer.analyze_monthly_patterns(data)
        
        if monthly_patterns is not None and len(monthly_patterns) > 0:
            # Statistical significance
            significance = statistical_analyzer.test_pattern_significance(monthly_patterns, 'day_of_month')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Day of month returns
                fig = visualizer.create_monthly_pattern_chart(monthly_patterns)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Key statistics
                first_half = monthly_patterns[monthly_patterns['day_of_month'] <= 15]['returns'].mean()
                second_half = monthly_patterns[monthly_patterns['day_of_month'] > 15]['returns'].mean()
                
                st.metric("First Half Month", f"{first_half:.4f}%")
                st.metric("Second Half Month", f"{second_half:.4f}%")
                
                p_value = significance.get('p_value', 1.0) if significance else 1.0
                is_significant = p_value < (1 - getattr(st.session_state, 'confidence_level', 0.95))
                st.metric("Significant Pattern", "Yes" if is_significant else "No")
            
            # Monthly statistics table
            if st.checkbox("Show Detailed Monthly Statistics"):
                monthly_stats = monthly_patterns.groupby('day_of_month')['returns'].agg([
                    'count', 'mean', 'std'
                ]).round(4)
                monthly_stats.columns = ['Observations', 'Avg Return (%)', 'Std Dev (%)']
                st.dataframe(monthly_stats)
        else:
            st.warning("Not enough data for monthly pattern analysis.")

def show_seasonal_analysis(data, seasonality_analyzer, visualizer, statistical_analyzer):
    """Display seasonal analysis with % gain from first trading day of year (choose alignment)"""
    st.header("🌟 Seasonal Patterns: % Gain from First Trading Day of Year")
    
    alignment = st.radio(
        "Alignment Mode",
        ["Trading day index (1..N)", "Calendar day-of-year (1..366)"],
        index=0,
        help="Trading day index aligns by trading sessions and ignores weekends/holidays. Calendar day-of-year aligns by calendar dates."
    )

    # Year filter for overlays
    all_years = sorted(data.index.year.unique())
    default_years = all_years[-10:] if len(all_years) > 10 else all_years
    selected_years = st.multiselect(
        "Include years",
        options=all_years,
        default=default_years
    )

    st.markdown("""
    Each year is normalized to 0% on its first trading day. The y-axis shows cumulative % gain from that baseline.
    """)
    
    with st.spinner("Calculating seasonal patterns..."):
        # Filter data for selected years
        data_sel = data[data.index.year.isin(selected_years)] if selected_years else data

        if alignment.startswith("Trading"):
            seasonal_results = seasonality_analyzer.analyze_seasonal_patterns(data_sel)
        else:
            seasonal_results = seasonality_analyzer.analyze_seasonal_patterns_calendar(data_sel)
        
        if seasonal_results is not None:
            # Show the main seasonal overlay chart
            if alignment.startswith("Trading"):
                fig = visualizer.create_seasonal_overlay_chart(seasonal_results)
            else:
                stride = st.slider("X-axis tick stride (days)", 1, 14, 7)
                fig = visualizer.create_calendar_overlay_chart(seasonal_results, day_tick_stride=stride)
            st.plotly_chart(fig, use_container_width=True)
            
            # Key statistics
            st.subheader("📊 Key Statistics")
            
            seasonal_avg = seasonal_results['seasonal_avg']
            data_results = seasonal_results['data']
            years = seasonal_results['years']

            col1, col2, col3, col4 = st.columns(4)
            
            if alignment.startswith("Trading"):
                with col1:
                    best_idx = int(seasonal_avg.loc[seasonal_avg['avg_gain'].idxmax(), 'trading_day'])
                    best_val = seasonal_avg['avg_gain'].max()
                    # Map to an example calendar date using first year present
                    try:
                        ref_year = int(min(years))
                        ref_df = data_results[data_results['year'] == ref_year]
                        ref_date = ref_df.loc[ref_df['trading_day'] == best_idx].index[0]
                        ref_label = ref_date.strftime('%b %d')
                    except Exception:
                        ref_label = ''
                    st.metric("Best Trading Day (index)", f"Day {best_idx} {('(' + ref_label + ')' ) if ref_label else ''}", f"{best_val:.2f}%")
                with col2:
                    worst_idx = int(seasonal_avg.loc[seasonal_avg['avg_gain'].idxmin(), 'trading_day'])
                    worst_val = seasonal_avg['avg_gain'].min()
                    try:
                        ref_year = int(min(years))
                        ref_df = data_results[data_results['year'] == ref_year]
                        ref_date = ref_df.loc[ref_df['trading_day'] == worst_idx].index[0]
                        ref_label = ref_date.strftime('%b %d')
                    except Exception:
                        ref_label = ''
                    st.metric("Worst Trading Day (index)", f"Day {worst_idx} {('(' + ref_label + ')' ) if ref_label else ''}", f"{worst_val:.2f}%")
                with col3:
                    max_days = seasonal_results.get('max_trading_days', seasonal_avg['trading_day'].max())
                    end_slice = seasonal_avg[seasonal_avg['trading_day'] >= (max_days - 5)]['avg_gain']
                    year_end_avg = end_slice.mean() if len(end_slice) > 0 else seasonal_avg['avg_gain'].iloc[-1]
                    st.metric("Avg Year-End Gain (approx)", f"{year_end_avg:.2f}%")
            else:
                with col1:
                    best_doy = int(seasonal_avg.loc[seasonal_avg['avg_gain'].idxmax(), 'day_of_year'])
                    best_val = seasonal_avg['avg_gain'].max()
                    # Convert DOY to label (non-leap year)
                    ref_date = datetime(2001, 1, 1) + timedelta(days=best_doy - 1)
                    st.metric("Best Calendar Day", ref_date.strftime('%b %d'), f"{best_val:.2f}%")
                with col2:
                    worst_doy = int(seasonal_avg.loc[seasonal_avg['avg_gain'].idxmin(), 'day_of_year'])
                    worst_val = seasonal_avg['avg_gain'].min()
                    ref_date = datetime(2001, 1, 1) + timedelta(days=worst_doy - 1)
                    st.metric("Worst Calendar Day", ref_date.strftime('%b %d'), f"{worst_val:.2f}%")
                with col3:
                    end_slice = seasonal_avg[seasonal_avg['day_of_year'] >= 360]['avg_gain']
                    year_end_avg = end_slice.mean() if len(end_slice) > 0 else seasonal_avg['avg_gain'].iloc[-1]
                    st.metric("Avg Year-End Gain (approx)", f"{year_end_avg:.2f}%")
            
            with col4:
                st.metric("Years Analyzed", f"{len(years)} years", f"{int(min(years))}-{int(max(years))}")

            # Optional detailed stats
            if st.checkbox("Show Detailed Day-by-Day Statistics"):
                if alignment.startswith("Trading"):
                    summary_stats = seasonal_avg.copy()
                    summary_stats['Trading Day'] = summary_stats['trading_day'].astype(int)
                    summary_stats = summary_stats[['Trading Day', 'avg_gain', 'std_gain', 'count']]
                    summary_stats.columns = ['Trading Day', 'Avg Gain (%)', 'Std Dev (%)', 'Observations']
                else:
                    summary_stats = seasonal_avg.copy()
                    summary_stats['Day of Year'] = summary_stats['day_of_year'].astype(int)
                    summary_stats = summary_stats[['Day of Year', 'avg_gain', 'std_gain', 'count']]
                    summary_stats.columns = ['Day of Year', 'Avg Gain (%)', 'Std Dev (%)', 'Observations']
                st.dataframe(summary_stats.round(3), use_container_width=True)
        else:
            st.warning("Not enough data for seasonal pattern analysis. Need at least 2 complete years of data.")

if __name__ == "__main__":
    main()
