"""
SPXW Option Explorer
Interactive Streamlit GUI for exploring individual SPXW option contracts
Analyzes Time & Sales data with comprehensive flow, price, and volume analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, time
from pathlib import Path
import glob
import re
import warnings

warnings.filterwarnings('ignore')

# Matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Constants
OPTION_MULTIPLIER = 100
INSTITUTIONAL_SIZE_THRESHOLD = 20  # Default value, can be adjusted in UI

# =====================================================
# DATA LOADING FUNCTIONS
# =====================================================

@st.cache_data
def get_available_dates():
    """Scan historical_data directory for available TAS files"""
    pattern = 'spxw_tas_data_*.csv'
    files = glob.glob(pattern)

    dates = []
    for file in files:
        # Extract date from filename: spxw_tas_data_YYYY-MM-DD.csv
        match = re.search(r'spxw_tas_data_(\d{4}-\d{2}-\d{2})\.csv', file)
        if match:
            dates.append(match.group(1))

    # Sort descending (most recent first)
    dates.sort(reverse=True)
    return dates


@st.cache_data
def load_tas_data(date):
    """Load and preprocess Time & Sales data for a specific date"""
    file_path = f'spxw_tas_data_{date}.csv'

    if not Path(file_path).exists():
        st.error(f"File not found: {file_path}")
        st.stop()

    # Load CSV
    df = pd.read_csv(file_path)

    # Validate required columns
    required_cols = ['date', 'time', 'symbol', 'price', 'size', 'aggressor_side']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns in data file: {missing}")
        st.stop()

    # Create datetime column
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')

    # Convert numeric columns
    numeric_cols = ['price', 'size', 'bid_price', 'ask_price', 'spx_price', 'spx_bid', 'spx_ask']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate notional value
    df['notional'] = df['price'] * df['size'] * OPTION_MULTIPLIER

    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)

    return df


# =====================================================
# OPTION SYMBOL FUNCTIONS
# =====================================================

def construct_option_symbol(strike, option_type, expiry_date):
    """Construct SPXW option symbol from components

    Args:
        strike: Strike price (e.g., 6775)
        option_type: 'CALL' or 'PUT'
        expiry_date: 'YYYY-MM-DD' format

    Returns:
        Symbol like '.SPXW251211C6775'
    """
    try:
        # Parse date
        dt = datetime.strptime(expiry_date, '%Y-%m-%d')
        yymmdd = dt.strftime('%y%m%d')

        # Convert option type to C/P
        cp = 'C' if option_type == 'CALL' else 'P'

        # Format strike as integer
        strike_str = str(int(strike))

        return f'.SPXW{yymmdd}{cp}{strike_str}'

    except ValueError as e:
        st.error(f"Invalid date format: {e}")
        return None


def parse_option_symbol(symbol):
    """Parse SPXW option symbol into components

    Pattern: \.([A-Z]+)(W)?(\d{6})([CP])(\d+)
    """
    match = re.match(r'\.([A-Z]+)(W)?(\d{6})([CP])(\d+)', str(symbol))
    if match:
        return {
            'ticker': match.group(1),
            'expiry_str': match.group(3),
            'strike': float(match.group(5)),
            'option_type': 'CALL' if match.group(4) == 'C' else 'PUT'
        }
    return None


# =====================================================
# FILTERING FUNCTIONS
# =====================================================

def get_available_expiries(df, strike, option_type):
    """Get all available expiries for a given strike and option type"""
    # Parse all symbols
    df['parsed_symbol'] = df['symbol'].apply(parse_option_symbol)
    df_parsed = df[df['parsed_symbol'].notna()].copy()

    # Extract strike and option type
    df_parsed['strike_val'] = df_parsed['parsed_symbol'].apply(lambda x: x['strike'])
    df_parsed['opt_type'] = df_parsed['parsed_symbol'].apply(lambda x: x['option_type'])

    # Filter by strike and option type
    filtered = df_parsed[
        (df_parsed['strike_val'] == strike) &
        (df_parsed['opt_type'] == option_type)
    ]

    # Get unique expiry strings and sort
    expiries = filtered['parsed_symbol'].apply(lambda x: x['expiry_str']).unique()
    expiries = sorted(expiries)

    return expiries


def filter_trades_by_symbol(df, symbol):
    """Filter trades for a specific option symbol"""
    filtered = df[df['symbol'] == symbol].copy()
    return filtered.reset_index(drop=True)


def filter_trades_by_strike_and_type(df, strike, option_type, expiry=None):
    """Filter trades by strike and option type, optionally filtered to specific expiry

    Args:
        df: TAS data
        strike: Strike price
        option_type: 'CALL' or 'PUT'
        expiry: Optional expiry string (YYMMDD format) or None for all expiries

    Returns:
        Filtered DataFrame with expiry_label column added
    """
    # Parse all symbols
    df['parsed_symbol'] = df['symbol'].apply(parse_option_symbol)
    df_parsed = df[df['parsed_symbol'].notna()].copy()

    # Extract components
    df_parsed['strike_val'] = df_parsed['parsed_symbol'].apply(lambda x: x['strike'])
    df_parsed['opt_type'] = df_parsed['parsed_symbol'].apply(lambda x: x['option_type'])
    df_parsed['expiry_str'] = df_parsed['parsed_symbol'].apply(lambda x: x['expiry_str'])

    # Filter by strike and option type
    filtered = df_parsed[
        (df_parsed['strike_val'] == strike) &
        (df_parsed['opt_type'] == option_type)
    ]

    # Optionally filter by expiry
    if expiry is not None and expiry != 'All':
        filtered = filtered[filtered['expiry_str'] == expiry]

    # Add readable expiry label (convert YYMMDD to YYYY-MM-DD)
    def format_expiry(expiry_str):
        try:
            dt = datetime.strptime(f"20{expiry_str}", "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except:
            return expiry_str

    filtered['expiry_label'] = filtered['expiry_str'].apply(format_expiry)

    return filtered.reset_index(drop=True)


def apply_additional_filters(df, min_size=0, time_range=None):
    """Apply optional filters (min trade size, time range)"""
    result = df.copy()

    # Size filter
    if min_size > 0:
        result = result[result['size'] >= min_size]

    # Time range filter
    if time_range is not None:
        start_time, end_time = time_range
        result['time_only'] = result['datetime'].dt.time
        result = result[
            (result['time_only'] >= start_time) &
            (result['time_only'] <= end_time)
        ]
        result = result.drop('time_only', axis=1)

    return result.reset_index(drop=True)


# =====================================================
# ANALYTICS FUNCTIONS
# =====================================================

def calculate_flow_metrics(df, institutional_threshold=None):
    """Calculate buy/sell flow metrics from aggressor_side column"""

    if institutional_threshold is None:
        institutional_threshold = INSTITUTIONAL_SIZE_THRESHOLD

    if len(df) == 0:
        return {}

    # Separate by aggressor side
    buy_trades = df[df['aggressor_side'] == 'BUY']
    sell_trades = df[df['aggressor_side'] == 'SELL']

    # Volume calculations
    buy_volume = buy_trades['size'].sum()
    sell_volume = sell_trades['size'].sum()
    total_volume = df['size'].sum()

    # Notional calculations
    buy_notional = buy_trades['notional'].sum()
    sell_notional = sell_trades['notional'].sum()
    total_notional = df['notional'].sum()

    # VWAP calculation
    vwap = (df['price'] * df['size']).sum() / df['size'].sum() if df['size'].sum() > 0 else 0

    # Buy percentage
    buy_pct = (buy_volume / total_volume * 100) if total_volume > 0 else 0

    # Institutional trades
    institutional_trades = len(df[df['size'] >= institutional_threshold])

    return {
        'total_trades': len(df),
        'total_volume': int(total_volume),
        'total_notional': float(total_notional),
        'buy_trades': len(buy_trades),
        'buy_volume': int(buy_volume),
        'buy_notional': float(buy_notional),
        'sell_trades': len(sell_trades),
        'sell_volume': int(sell_volume),
        'sell_notional': float(sell_notional),
        'net_volume': int(buy_volume - sell_volume),
        'net_notional': float(buy_notional - sell_notional),
        'buy_pct': float(buy_pct),
        'vwap': float(vwap),
        'avg_trade_size': float(df['size'].mean()),
        'max_trade_size': int(df['size'].max()),
        'min_price': float(df['price'].min()),
        'max_price': float(df['price'].max()),
        'price_range': float(df['price'].max() - df['price'].min()),
        'institutional_trades': institutional_trades
    }


def calculate_time_series_data(df, window_minutes=5):
    """Aggregate trades into time windows for charting"""

    if len(df) == 0:
        return pd.DataFrame()

    # Create time windows
    df_copy = df.copy()
    df_copy['time_window'] = df_copy['datetime'].dt.floor(f'{window_minutes}min')

    # Separate buy/sell
    buy_df = df_copy[df_copy['aggressor_side'] == 'BUY'].copy()
    sell_df = df_copy[df_copy['aggressor_side'] == 'SELL'].copy()

    # Aggregate by time window
    buy_by_window = buy_df.groupby('time_window')['size'].sum()
    sell_by_window = sell_df.groupby('time_window')['size'].sum()

    # VWAP by window
    vwap_by_window = df_copy.groupby('time_window').apply(
        lambda x: (x['price'] * x['size']).sum() / x['size'].sum() if x['size'].sum() > 0 else 0
    )

    # Trade count
    count_by_window = df_copy.groupby('time_window').size()

    # Create time series DataFrame
    all_windows = pd.DataFrame(index=pd.date_range(
        df_copy['time_window'].min(),
        df_copy['time_window'].max(),
        freq=f'{window_minutes}min'
    ))

    time_series = pd.DataFrame({
        'time_window': all_windows.index,
        'buy_volume': all_windows.index.map(lambda x: buy_by_window.get(x, 0)),
        'sell_volume': all_windows.index.map(lambda x: sell_by_window.get(x, 0)),
        'vwap': all_windows.index.map(lambda x: vwap_by_window.get(x, 0)),
        'trade_count': all_windows.index.map(lambda x: count_by_window.get(x, 0))
    })

    # Calculate cumulative sums
    time_series['cumulative_buy'] = time_series['buy_volume'].cumsum()
    time_series['cumulative_sell'] = time_series['sell_volume'].cumsum()
    time_series['total_volume'] = time_series['buy_volume'] + time_series['sell_volume']

    return time_series


def identify_institutional_trades(df, size_threshold=INSTITUTIONAL_SIZE_THRESHOLD):
    """Filter trades with size >= threshold"""
    return df[df['size'] >= size_threshold].sort_values('size', ascending=False).reset_index(drop=True)


# =====================================================
# SPREAD DETECTION FUNCTIONS
# =====================================================

def parse_option_symbol_full(symbol):
    """
    Parse SPXW option symbol with full component extraction

    Args:
        symbol: Option symbol (e.g., '.SPXW251211C6880')

    Returns:
        dict with keys: ticker, expiry_str, expiry_date, strike, option_type
        Returns None if parsing fails
    """
    match = re.match(r'\.([A-Z]+)(W)?(\d{6})([CP])(\d+)', str(symbol))
    if not match:
        return None

    expiry_str = match.group(3)  # YYMMDD
    try:
        expiry_date = datetime.strptime(f"20{expiry_str}", "%Y%m%d").date()
    except:
        expiry_date = None

    return {
        'ticker': match.group(1),
        'expiry_str': expiry_str,
        'expiry_date': expiry_date,
        'strike': float(match.group(5)),
        'option_type': 'C' if match.group(4) == 'C' else 'P'
    }


def identify_2leg_spread(trade1, trade2):
    """
    Identify 2-leg spread patterns (vertical, calendar)

    Returns dict with spread info or None if no pattern matched
    """

    # Size matching (1:1, 2:1, or 1:2 ratios)
    size_ratio = trade1['size'] / trade2['size']
    if not (0.45 <= size_ratio <= 2.1):  # Allow 1:2 to 2:1
        return None

    # Check if same side or opposite sides
    same_side = trade1['aggressor_side'] == trade2['aggressor_side']

    if same_side:
        # Same side spread (both BUY or both SELL)
        # Typical for short strangles, credit spreads, etc.
        leg1 = trade1
        leg2 = trade2
        spread_direction = trade1['aggressor_side']  # Both are same side
    else:
        # Opposite sides - traditional vertical spread
        # Determine which is bought and which is sold
        buy_leg = trade1 if trade1['aggressor_side'] == 'BUY' else trade2
        sell_leg = trade2 if trade1['aggressor_side'] == 'BUY' else trade1
        leg1 = buy_leg
        leg2 = sell_leg
        spread_direction = 'OPPOSITE'

    # Same expiry, different strike, same option type → VERTICAL SPREAD
    if (leg1['expiry_str'] == leg2['expiry_str'] and
        leg1['strike'] != leg2['strike'] and
        leg1['opt_type'] == leg2['opt_type']):

        min_size = min(leg1['size'], leg2['size'])

        # Calculate net debit/credit based on whether same or opposite sides
        if same_side:
            # Both same side: net is total premium (positive for SELL, negative for BUY)
            total_premium = (leg1['price'] + leg2['price']) * min_size * OPTION_MULTIPLIER
            net_debit_credit = total_premium if spread_direction == 'SELL' else -total_premium
            spread_subtype = f"{spread_direction.lower()}_vertical"
        else:
            # Opposite sides: traditional vertical spread
            net_debit_credit = (leg1['price'] - leg2['price']) * min_size * OPTION_MULTIPLIER
            spread_subtype = "vertical"

        return {
            'spread_type': spread_subtype,
            'timestamp': leg1['datetime'],
            'option_type': 'CALL' if leg1['opt_type'] == 'C' else 'PUT',
            'expiries': [leg1['expiry_str']],
            'strikes': sorted([leg1['strike'], leg2['strike']]),
            'total_size': min_size,
            'net_debit_credit': net_debit_credit,
            'total_notional': abs(leg1['notional']) + abs(leg2['notional']),
            'leg_count': 2,
            'spread_direction': spread_direction,
            'legs': [
                {'side': leg1['aggressor_side'], 'strike': leg1['strike'], 'price': leg1['price'], 'size': leg1['size']},
                {'side': leg2['aggressor_side'], 'strike': leg2['strike'], 'price': leg2['price'], 'size': leg2['size']}
            ]
        }

    # Different expiry, same strike, same option type → CALENDAR SPREAD
    elif (leg1['expiry_str'] != leg2['expiry_str'] and
          leg1['strike'] == leg2['strike'] and
          leg1['opt_type'] == leg2['opt_type']):

        min_size = min(leg1['size'], leg2['size'])

        # Calculate net debit/credit based on whether same or opposite sides
        if same_side:
            total_premium = (leg1['price'] + leg2['price']) * min_size * OPTION_MULTIPLIER
            net_debit_credit = total_premium if spread_direction == 'SELL' else -total_premium
            spread_subtype = f"{spread_direction.lower()}_calendar"
        else:
            net_debit_credit = (leg1['price'] - leg2['price']) * min_size * OPTION_MULTIPLIER
            spread_subtype = "calendar"

        return {
            'spread_type': spread_subtype,
            'timestamp': leg1['datetime'],
            'option_type': 'CALL' if leg1['opt_type'] == 'C' else 'PUT',
            'expiries': sorted([leg1['expiry_str'], leg2['expiry_str']]),
            'strikes': [leg1['strike']],
            'total_size': min_size,
            'net_debit_credit': net_debit_credit,
            'total_notional': abs(leg1['notional']) + abs(leg2['notional']),
            'leg_count': 2,
            'spread_direction': spread_direction,
            'legs': [
                {'side': leg1['aggressor_side'], 'expiry': leg1['expiry_str'], 'price': leg1['price'], 'size': leg1['size']},
                {'side': leg2['aggressor_side'], 'expiry': leg2['expiry_str'], 'price': leg2['price'], 'size': leg2['size']}
            ]
        }

    return None


def detect_spreads(df, time_window_seconds=5):
    """
    Detect multi-leg option spreads using tight time windows

    Args:
        df: TAS data with datetime, symbol, size, aggressor_side, sequence
        time_window_seconds: Maximum time between spread legs (default 5s)

    Returns:
        DataFrame with detected spreads
    """

    # Filter for institutional size for performance (≥10 contracts)
    df_inst = df[df['size'] >= 10].copy()

    if len(df_inst) < 2:
        return pd.DataFrame()

    # Parse all symbols upfront
    df_inst['parsed'] = df_inst['symbol'].apply(parse_option_symbol_full)
    df_inst = df_inst[df_inst['parsed'].notna()].copy()

    # Extract components
    df_inst['expiry_str'] = df_inst['parsed'].apply(lambda x: x['expiry_str'])
    df_inst['expiry_date'] = df_inst['parsed'].apply(lambda x: x['expiry_date'])
    df_inst['strike'] = df_inst['parsed'].apply(lambda x: x['strike'])
    df_inst['opt_type'] = df_inst['parsed'].apply(lambda x: x['option_type'])

    # Sort by timestamp and sequence
    df_inst = df_inst.sort_values(['datetime', 'sequence'])

    spreads = []
    spread_id = 0

    # Group by time buckets (5 second windows)
    df_inst['time_bucket'] = df_inst['datetime'].dt.floor(f'{time_window_seconds}s')

    for bucket_time, bucket_df in df_inst.groupby('time_bucket'):
        if len(bucket_df) < 2:
            continue

        # Convert to list for easier iteration
        trades = bucket_df.to_dict('records')

        # Try to match spreads
        for i in range(len(trades)):
            for j in range(i + 1, min(i + 20, len(trades))):  # Limit lookahead for performance
                trade1 = trades[i]
                trade2 = trades[j]

                # Time proximity check (within 5s)
                time_diff = abs((trade2['datetime'] - trade1['datetime']).total_seconds())
                if time_diff > time_window_seconds:
                    continue

                # Sequence proximity (within 50 sequence numbers)
                if 'sequence' in trade1 and 'sequence' in trade2:
                    seq_diff = abs(trade2['sequence'] - trade1['sequence'])
                    if seq_diff > 50:
                        continue

                # Check for 2-leg spread patterns
                spread_info = identify_2leg_spread(trade1, trade2)
                if spread_info:
                    spread_info['spread_id'] = spread_id
                    spreads.append(spread_info)
                    spread_id += 1

    if not spreads:
        return pd.DataFrame()

    return pd.DataFrame(spreads)


def calculate_spread_metrics(spread_df):
    """Calculate aggregate metrics for detected spreads"""

    if len(spread_df) == 0:
        return {
            'total_spreads': 0,
            'by_type': {},
            'total_notional': 0,
            'avg_spread_width': 0,
            'total_debit': 0,
            'total_credit': 0
        }

    # Count by type
    by_type = spread_df['spread_type'].value_counts().to_dict()

    # Total notional
    total_notional = spread_df['total_notional'].sum()

    # Average spread width (for vertical spreads only)
    vertical_spreads = spread_df[spread_df['spread_type'] == 'vertical']
    if len(vertical_spreads) > 0:
        vertical_spreads = vertical_spreads.copy()
        vertical_spreads['spread_width'] = vertical_spreads['strikes'].apply(
            lambda x: max(x) - min(x) if isinstance(x, list) and len(x) == 2 else 0
        )
        avg_spread_width = vertical_spreads['spread_width'].mean()
    else:
        avg_spread_width = 0

    # Debit vs credit
    total_debit = spread_df[spread_df['net_debit_credit'] > 0]['net_debit_credit'].sum()
    total_credit = abs(spread_df[spread_df['net_debit_credit'] < 0]['net_debit_credit'].sum())

    return {
        'total_spreads': len(spread_df),
        'by_type': by_type,
        'total_notional': total_notional,
        'avg_spread_width': avg_spread_width,
        'total_debit': total_debit,
        'total_credit': total_credit
    }


# =====================================================
# INSTITUTIONAL TIMING ANALYSIS FUNCTIONS
# =====================================================

def analyze_hourly_flow(df, institutional_threshold=100):
    """Aggregate institutional trade flow by hour"""

    # Filter institutional trades
    df_inst = df[df['size'] >= institutional_threshold].copy()

    if len(df_inst) == 0:
        return pd.DataFrame()

    # Extract hour
    df_inst['hour'] = df_inst['datetime'].dt.hour

    # Separate buy/sell
    buy_df = df_inst[df_inst['aggressor_side'] == 'BUY']
    sell_df = df_inst[df_inst['aggressor_side'] == 'SELL']

    # Aggregate by hour
    hourly = pd.DataFrame()

    for hour in range(9, 17):  # 9 AM to 4 PM
        hour_data = {
            'hour': hour,
            'hour_label': f"{hour}:00-{hour}:59",
            'buy_volume': buy_df[buy_df['hour'] == hour]['size'].sum(),
            'sell_volume': sell_df[sell_df['hour'] == hour]['size'].sum(),
            'buy_notional': buy_df[buy_df['hour'] == hour]['notional'].sum(),
            'sell_notional': sell_df[sell_df['hour'] == hour]['notional'].sum(),
            'trade_count': len(df_inst[df_inst['hour'] == hour])
        }

        hourly = pd.concat([hourly, pd.DataFrame([hour_data])], ignore_index=True)

    # Calculate net
    hourly['net_volume'] = hourly['buy_volume'] - hourly['sell_volume']
    hourly['net_notional'] = hourly['buy_notional'] - hourly['sell_notional']

    return hourly


def analyze_30min_windows(df, institutional_threshold=100):
    """Aggregate institutional flow into 30-minute windows"""

    # Filter institutional
    df_inst = df[df['size'] >= institutional_threshold].copy()

    if len(df_inst) == 0:
        return pd.DataFrame()

    # Create 30-minute windows
    df_inst['window_start'] = df_inst['datetime'].dt.floor('30min')

    # Separate buy/sell for aggregation
    buy_mask = df_inst['aggressor_side'] == 'BUY'
    sell_mask = df_inst['aggressor_side'] == 'SELL'

    # Aggregate
    windows = df_inst.groupby('window_start').agg(
        buy_volume=('size', lambda x: x[buy_mask.loc[x.index]].sum()),
        sell_volume=('size', lambda x: x[sell_mask.loc[x.index]].sum()),
        buy_notional=('notional', lambda x: x[buy_mask.loc[x.index]].sum()),
        sell_notional=('notional', lambda x: x[sell_mask.loc[x.index]].sum()),
        trade_count=('size', 'count')
    ).reset_index()

    # Calculate net
    windows['net_volume'] = windows['buy_volume'] - windows['sell_volume']
    windows['net_notional'] = windows['buy_notional'] - windows['sell_notional']

    # Create labels
    windows['window_label'] = windows['window_start'].dt.strftime('%H:%M')

    return windows


def analyze_market_periods(df, institutional_threshold=100):
    """Analyze institutional flow by market period"""

    # Filter institutional
    df_inst = df[df['size'] >= institutional_threshold].copy()

    if len(df_inst) == 0:
        return {
            'opening': {'buy_volume': 0, 'sell_volume': 0, 'net_volume': 0, 'trade_count': 0, 'avg_trade_size': 0, 'buy_notional': 0, 'sell_notional': 0},
            'midday': {'buy_volume': 0, 'sell_volume': 0, 'net_volume': 0, 'trade_count': 0, 'avg_trade_size': 0, 'buy_notional': 0, 'sell_notional': 0},
            'power_hour': {'buy_volume': 0, 'sell_volume': 0, 'net_volume': 0, 'trade_count': 0, 'avg_trade_size': 0, 'buy_notional': 0, 'sell_notional': 0}
        }

    # Extract time
    df_inst['time_only'] = df_inst['datetime'].dt.time

    # Define periods
    periods = {
        'opening': (time(9, 30), time(10, 30)),
        'midday': (time(12, 0), time(14, 0)),
        'power_hour': (time(15, 0), time(16, 0))
    }

    results = {}

    for period_name, (start_time, end_time) in periods.items():
        period_df = df_inst[
            (df_inst['time_only'] >= start_time) &
            (df_inst['time_only'] <= end_time)
        ]

        buy_vol = period_df[period_df['aggressor_side'] == 'BUY']['size'].sum()
        sell_vol = period_df[period_df['aggressor_side'] == 'SELL']['size'].sum()

        results[period_name] = {
            'buy_volume': int(buy_vol),
            'sell_volume': int(sell_vol),
            'net_volume': int(buy_vol - sell_vol),
            'trade_count': len(period_df),
            'avg_trade_size': float(period_df['size'].mean()) if len(period_df) > 0 else 0,
            'buy_notional': float(period_df[period_df['aggressor_side'] == 'BUY']['notional'].sum()),
            'sell_notional': float(period_df[period_df['aggressor_side'] == 'SELL']['notional'].sum())
        }

    return results


# =====================================================
# CHART FUNCTIONS
# =====================================================

def plot_flow_breakdown(metrics):
    """Pie chart showing buy vs sell volume distribution"""
    fig, ax = plt.subplots(figsize=(8, 6))

    sizes = [metrics['buy_volume'], metrics['sell_volume']]
    labels = [f"BUY\n{metrics['buy_volume']:,}", f"SELL\n{metrics['sell_volume']:,}"]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)

    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           explode=explode, shadow=True, startangle=90)
    ax.set_title('Buy vs Sell Volume Distribution', fontsize=14, fontweight='bold')

    return fig


def plot_time_series_flow(time_series):
    """Stacked area chart showing buy/sell volume over time"""
    fig, ax = plt.subplots(figsize=(14, 6))

    times = time_series['time_window']

    ax.fill_between(times, 0, time_series['buy_volume'],
                    alpha=0.7, color='green', label='BUY Volume')
    ax.fill_between(times, 0, -time_series['sell_volume'],
                    alpha=0.7, color='red', label='SELL Volume')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Contract Volume')
    ax.set_title('Intraday Flow by Aggressor Side', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)

    return fig


def plot_net_flow(time_series):
    """Bar chart showing net flow (buy - sell) over time"""
    fig, ax = plt.subplots(figsize=(14, 6))

    net_flow = time_series['buy_volume'] - time_series['sell_volume']
    colors = ['green' if x > 0 else 'red' for x in net_flow]

    ax.bar(time_series['time_window'], net_flow,
           width=0.002, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Net Flow (Contracts)')
    ax.set_title('Net Directional Flow', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)

    return fig


def plot_price_time_series(df, time_series):
    """Scatter plot of individual trades with VWAP overlay"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Scatter individual trades
    buy_trades = df[df['aggressor_side'] == 'BUY']
    sell_trades = df[df['aggressor_side'] == 'SELL']

    if len(buy_trades) > 0:
        ax.scatter(buy_trades['datetime'], buy_trades['price'],
                  c='green', alpha=0.4, s=buy_trades['size']*2, label='BUY')
    if len(sell_trades) > 0:
        ax.scatter(sell_trades['datetime'], sell_trades['price'],
                  c='red', alpha=0.4, s=sell_trades['size']*2, label='SELL')

    # VWAP line
    if len(time_series) > 0:
        vwap_data = time_series[time_series['vwap'] > 0]
        if len(vwap_data) > 0:
            ax.plot(vwap_data['time_window'], vwap_data['vwap'],
                   color='blue', linewidth=2, label='VWAP', linestyle='--')

    ax.set_xlabel('Time')
    ax.set_ylabel('Price ($)')
    ax.set_title('Price Action with VWAP (Bubble size = trade size)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)

    return fig


def plot_cumulative_volume(time_series):
    """Line chart showing cumulative buy/sell volume throughout the day"""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(time_series['time_window'], time_series['cumulative_buy'],
           color='green', linewidth=2, label='Cumulative BUY', alpha=0.8)
    ax.plot(time_series['time_window'], time_series['cumulative_sell'],
           color='red', linewidth=2, label='Cumulative SELL', alpha=0.8)

    net_cumulative = time_series['cumulative_buy'] - time_series['cumulative_sell']
    ax.plot(time_series['time_window'], net_cumulative,
           color='blue', linewidth=2.5, label='Net Cumulative', linestyle='--')

    ax.fill_between(time_series['time_window'], 0, net_cumulative,
                   where=(net_cumulative >= 0), alpha=0.2, color='green')
    ax.fill_between(time_series['time_window'], 0, net_cumulative,
                   where=(net_cumulative < 0), alpha=0.2, color='red')

    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Volume')
    ax.set_title('Cumulative Flow Throughout Trading Day', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)

    return fig


def plot_size_distribution(df):
    """Histogram showing distribution of trade sizes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Full distribution
    ax1.hist(df['size'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Trade Size (contracts)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Trade Size Distribution', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Log scale for better visibility of large trades
    ax2.hist(df['size'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Trade Size (contracts)')
    ax2.set_ylabel('Frequency (log scale)')
    ax2.set_title('Trade Size Distribution (Log Scale)', fontsize=11, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_institutional_timeline(institutional_df, threshold=None):
    """Scatter plot showing when institutional trades occurred"""
    if threshold is None:
        threshold = INSTITUTIONAL_SIZE_THRESHOLD

    fig, ax = plt.subplots(figsize=(14, 6))

    buy_inst = institutional_df[institutional_df['aggressor_side'] == 'BUY']
    sell_inst = institutional_df[institutional_df['aggressor_side'] == 'SELL']

    if len(buy_inst) > 0:
        ax.scatter(buy_inst['datetime'], buy_inst['size'],
                  c='green', alpha=0.6, s=buy_inst['size']*3,
                  marker='^', label=f'BUY (≥{threshold} contracts)')
    if len(sell_inst) > 0:
        ax.scatter(sell_inst['datetime'], sell_inst['size'],
                  c='red', alpha=0.6, s=sell_inst['size']*3,
                  marker='v', label=f'SELL (≥{threshold} contracts)')

    ax.set_xlabel('Time')
    ax.set_ylabel('Trade Size (contracts)')
    ax.set_title('Institutional-Sized Trades Timeline', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)

    return fig


# =====================================================
# SPREAD ANALYSIS CHARTS
# =====================================================

def plot_spread_type_breakdown(spread_df):
    """Pie chart showing distribution of spread types"""
    fig, ax = plt.subplots(figsize=(8, 6))

    if len(spread_df) == 0:
        ax.text(0.5, 0.5, 'No spreads detected',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    type_counts = spread_df['spread_type'].value_counts()

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    explode = [0.05] * len(type_counts)

    ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
           colors=colors, explode=explode, shadow=True, startangle=90)
    ax.set_title('Spread Types Distribution', fontsize=14, fontweight='bold')

    return fig


def plot_spread_count_by_hour(spread_df):
    """Bar chart showing spread detection by hour"""
    fig, ax = plt.subplots(figsize=(12, 6))

    if len(spread_df) == 0:
        ax.text(0.5, 0.5, 'No spreads detected',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    spread_df_copy = spread_df.copy()
    spread_df_copy['hour'] = pd.to_datetime(spread_df_copy['timestamp']).dt.hour

    hourly_counts = spread_df_copy.groupby(['hour', 'spread_type']).size().unstack(fill_value=0)

    hourly_counts.plot(kind='bar', stacked=True, ax=ax,
                       colormap='Set3', width=0.7)

    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('Spread Count', fontsize=11)
    ax.set_title('Spread Detection by Hour', fontsize=12, fontweight='bold')
    ax.legend(title='Spread Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=0)
    plt.tight_layout()

    return fig


def plot_cumulative_spreads(spread_df):
    """Line chart showing cumulative spread detection throughout day"""
    fig, ax = plt.subplots(figsize=(14, 6))

    if len(spread_df) == 0:
        ax.text(0.5, 0.5, 'No spreads detected',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    spread_df_sorted = spread_df.sort_values('timestamp').copy()
    spread_df_sorted['cumulative_count'] = range(1, len(spread_df_sorted) + 1)

    # Plot cumulative by type
    for spread_type in spread_df_sorted['spread_type'].unique():
        type_df = spread_df_sorted[spread_df_sorted['spread_type'] == spread_type].copy()
        type_df['type_cumulative'] = range(1, len(type_df) + 1)
        ax.plot(type_df['timestamp'], type_df['type_cumulative'],
                label=spread_type.capitalize(), linewidth=2, marker='o', markersize=3, alpha=0.7)

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Cumulative Spread Count', fontsize=11)
    ax.set_title('Cumulative Spread Detection Throughout Trading Day', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


# =====================================================
# INSTITUTIONAL TIMING CHARTS
# =====================================================

def plot_hourly_institutional_volume(hourly_df, threshold=None):
    """Stacked bar chart showing hourly institutional buy/sell volume"""
    if threshold is None:
        threshold = INSTITUTIONAL_SIZE_THRESHOLD

    fig, ax = plt.subplots(figsize=(14, 7))

    if len(hourly_df) == 0:
        ax.text(0.5, 0.5, 'No institutional trades found',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    x = np.arange(len(hourly_df))
    width = 0.35

    buy_bars = ax.bar(x - width/2, hourly_df['buy_volume'], width,
                      label='BUY', color='green', alpha=0.8)
    sell_bars = ax.bar(x + width/2, hourly_df['sell_volume'], width,
                       label='SELL', color='red', alpha=0.8)

    ax.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Volume (Contracts)', fontsize=11, fontweight='bold')
    ax.set_title(f'Hourly Institutional Volume (≥{threshold} contracts)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(hourly_df['hour_label'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [buy_bars, sell_bars]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig


def plot_30min_heatmap(windows_df):
    """Heatmap showing institutional flow in 30-minute windows"""
    fig, ax = plt.subplots(figsize=(14, 8))

    if len(windows_df) == 0:
        ax.text(0.5, 0.5, 'No institutional trades found',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Create pivot for heatmap: rows = metric, columns = time window
    heatmap_data = windows_df[['window_label', 'buy_volume', 'sell_volume', 'net_volume']].set_index('window_label').T

    # Plot heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='RdYlGn',
                center=0, cbar_kws={'label': 'Volume'}, ax=ax, linewidths=0.5)

    ax.set_title('Institutional Flow by 30-Minute Windows', fontsize=13, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=11)
    ax.set_xlabel('Time Window', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig


def plot_market_period_comparison(period_results):
    """Bar chart comparing institutional flow across market periods"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    periods = ['opening', 'midday', 'power_hour']
    period_labels = ['Opening\n(9:30-10:30)', 'Midday\n(12:00-14:00)', 'Power Hour\n(15:00-16:00)']

    # Volume comparison
    buy_vols = [period_results[p]['buy_volume'] for p in periods]
    sell_vols = [period_results[p]['sell_volume'] for p in periods]

    x = np.arange(len(periods))
    width = 0.35

    ax1.bar(x - width/2, buy_vols, width, label='BUY', color='green', alpha=0.8)
    ax1.bar(x + width/2, sell_vols, width, label='SELL', color='red', alpha=0.8)
    ax1.set_ylabel('Volume (Contracts)', fontsize=11, fontweight='bold')
    ax1.set_title('Volume by Market Period', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(period_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Trade count comparison
    trade_counts = [period_results[p]['trade_count'] for p in periods]
    ax2.bar(period_labels, trade_counts, color='steelblue', alpha=0.8)
    ax2.set_ylabel('Trade Count', fontsize=11, fontweight='bold')
    ax2.set_title('Institutional Trade Count by Period', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (b, s) in enumerate(zip(buy_vols, sell_vols)):
        if b > 0:
            ax1.text(i - width/2, b, f'{int(b):,}', ha='center', va='bottom', fontsize=9)
        if s > 0:
            ax1.text(i + width/2, s, f'{int(s):,}', ha='center', va='bottom', fontsize=9)

    for i, tc in enumerate(trade_counts):
        if tc > 0:
            ax2.text(i, tc, f'{int(tc):,}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def plot_top_strikes_chart(top_strikes_df):
    """Horizontal bar chart showing top strikes by volume"""
    fig, ax = plt.subplots(figsize=(12, 6))

    if len(top_strikes_df) == 0:
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    # Create labels
    labels = top_strikes_df.apply(
        lambda row: f"{row['option_type']} {int(row['strike'])} ({row['expiry_label']})",
        axis=1
    ).tolist()

    # Reverse order for plotting (top to bottom)
    labels = labels[::-1]
    volumes = top_strikes_df['total_volume'].tolist()[::-1]

    # Color by option type
    colors = ['green' if 'CALL' in label else 'red' for label in labels]

    # Create horizontal bar chart
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, volumes, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for i, (bar, vol) in enumerate(zip(bars, volumes)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f' {int(vol):,}',
                ha='left', va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Volume (Contracts)', fontsize=11, fontweight='bold')
    ax.set_title('Top 10 Strikes by Volume', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig


def plot_institutional_cumulative_volume(df, institutional_threshold=100):
    """Line chart showing cumulative institutional volume over time"""
    fig, ax = plt.subplots(figsize=(14, 6))

    df_inst = df[df['size'] >= institutional_threshold].copy()

    if len(df_inst) == 0:
        ax.text(0.5, 0.5, 'No institutional trades found',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    df_inst = df_inst.sort_values('datetime')

    # Separate buy/sell
    buy_df = df_inst[df_inst['aggressor_side'] == 'BUY'].copy()
    sell_df = df_inst[df_inst['aggressor_side'] == 'SELL'].copy()

    buy_df['cumulative_volume'] = buy_df['size'].cumsum()
    sell_df['cumulative_volume'] = sell_df['size'].cumsum()

    ax.plot(buy_df['datetime'], buy_df['cumulative_volume'],
            color='green', linewidth=2, label='Cumulative BUY', alpha=0.8)
    ax.plot(sell_df['datetime'], sell_df['cumulative_volume'],
            color='red', linewidth=2, label='Cumulative SELL', alpha=0.8)

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Cumulative Volume (Contracts)', fontsize=11)
    ax.set_title('Cumulative Institutional Volume Throughout Day', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


# =====================================================
# DAILY OVERVIEW FUNCTIONS
# =====================================================

def analyze_top_strikes(df, top_n=10):
    """Analyze the full day's data to find top strikes by volume and notional

    Returns:
        DataFrame with top strikes sorted by volume
    """
    # Parse all symbols
    df['parsed_symbol'] = df['symbol'].apply(parse_option_symbol)
    df_parsed = df[df['parsed_symbol'].notna()].copy()

    # Extract components
    df_parsed['strike'] = df_parsed['parsed_symbol'].apply(lambda x: x['strike'])
    df_parsed['option_type'] = df_parsed['parsed_symbol'].apply(lambda x: x['option_type'])
    df_parsed['expiry_str'] = df_parsed['parsed_symbol'].apply(lambda x: x['expiry_str'])

    # Group by strike, option_type, and expiry
    grouped = df_parsed.groupby(['strike', 'option_type', 'expiry_str']).agg({
        'size': 'sum',
        'notional': 'sum',
        'price': lambda x: (x * df_parsed.loc[x.index, 'size']).sum() / df_parsed.loc[x.index, 'size'].sum(),  # VWAP
        'datetime': 'count'
    }).reset_index()

    grouped.columns = ['strike', 'option_type', 'expiry_str', 'total_volume', 'total_notional', 'vwap', 'trade_count']

    # Format expiry for display
    def format_expiry(expiry_str):
        try:
            dt = datetime.strptime(f"20{expiry_str}", "%Y%m%d")
            return dt.strftime("%m/%d")
        except:
            return expiry_str

    grouped['expiry_label'] = grouped['expiry_str'].apply(format_expiry)

    # Sort by volume
    grouped = grouped.sort_values('total_volume', ascending=False)

    return grouped.head(top_n)


def analyze_call_put_flow(df):
    """Analyze overall call vs put flow for the day"""
    # Parse all symbols
    df['parsed_symbol'] = df['symbol'].apply(parse_option_symbol)
    df_parsed = df[df['parsed_symbol'].notna()].copy()

    # Extract option type
    df_parsed['option_type'] = df_parsed['parsed_symbol'].apply(lambda x: x['option_type'])

    # Aggregate by option type and aggressor side
    summary = df_parsed.groupby(['option_type', 'aggressor_side']).agg({
        'size': 'sum',
        'notional': 'sum'
    }).reset_index()

    return summary


def analyze_expiry_breakdown(df):
    """Analyze volume and notional by expiry date"""
    # Parse all symbols
    df['parsed_symbol'] = df['symbol'].apply(parse_option_symbol)
    df_parsed = df[df['parsed_symbol'].notna()].copy()

    # Extract expiry
    df_parsed['expiry_str'] = df_parsed['parsed_symbol'].apply(lambda x: x['expiry_str'])

    # Aggregate by expiry
    expiry_summary = df_parsed.groupby('expiry_str').agg({
        'size': 'sum',
        'notional': 'sum',
        'datetime': 'count'
    }).reset_index()

    expiry_summary.columns = ['expiry_str', 'total_volume', 'total_notional', 'trade_count']

    # Format expiry for display
    def format_expiry(expiry_str):
        try:
            dt = datetime.strptime(f"20{expiry_str}", "%Y%m%d")
            return dt.strftime("%Y-%m-%d (%a)")
        except:
            return expiry_str

    expiry_summary['expiry_label'] = expiry_summary['expiry_str'].apply(format_expiry)

    # Sort by expiry date
    expiry_summary = expiry_summary.sort_values('expiry_str')

    return expiry_summary


# =====================================================
# UI COMPONENTS
# =====================================================

def render_sidebar():
    """Render sidebar with all input controls"""

    st.sidebar.title("SPXW Option Explorer")
    st.sidebar.markdown("---")

    # Date selector
    dates = get_available_dates()
    if not dates:
        st.sidebar.error("No TAS data files found!")
        st.stop()

    selected_date = st.sidebar.selectbox(
        "Trading Date",
        options=dates,
        index=0,
        help="Select trading date to analyze"
    )

    # Strike input
    strike = st.sidebar.number_input(
        "Strike Price",
        min_value=1000,
        max_value=10000,
        value=6800,
        step=5,
        help="Enter strike price (e.g., 6800)"
    )

    # Call/Put selector
    option_type = st.sidebar.radio(
        "Option Type",
        options=["CALL", "PUT"],
        index=0,
        horizontal=True
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters (Optional)")

    # Institutional threshold
    institutional_threshold = st.sidebar.slider(
        "Institutional Size Threshold",
        min_value=5,
        max_value=200,
        value=20,
        step=5,
        help="Minimum contract size to classify as institutional trade"
    )

    # Min size filter
    min_size = st.sidebar.slider(
        "Min Trade Size",
        min_value=0,
        max_value=500,
        value=0,
        step=10,
        help="Filter trades by minimum contract size"
    )

    # Time range filter
    use_time_filter = st.sidebar.checkbox("Filter by Time Range")
    time_range = None
    if use_time_filter:
        col1, col2 = st.sidebar.columns(2)
        start_time = col1.time_input("Start", value=time(9, 30))
        end_time = col2.time_input("End", value=time(16, 0))
        time_range = (start_time, end_time)

    return {
        'selected_date': selected_date,
        'strike': strike,
        'option_type': option_type,
        'min_size': min_size,
        'time_range': time_range,
        'institutional_threshold': institutional_threshold
    }


def render_metrics_overview(metrics, label):
    """Render key metrics in a clean layout"""

    st.subheader(f"{label} - Trading Summary")

    # Row 1: Volume metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Trades", f"{metrics['total_trades']:,}")
    col2.metric("Total Volume", f"{metrics['total_volume']:,}")
    col3.metric("Total Notional", f"${metrics['total_notional']:,.0f}")
    col4.metric("VWAP", f"${metrics['vwap']:.2f}")

    st.markdown("---")

    # Row 2: Buy/Sell breakdown
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "BUY Volume",
        f"{metrics['buy_volume']:,}",
        delta=f"{metrics['buy_pct']:.1f}% of total"
    )

    col2.metric(
        "SELL Volume",
        f"{metrics['sell_volume']:,}",
        delta=f"{100-metrics['buy_pct']:.1f}% of total"
    )

    col3.metric(
        "Net Flow",
        f"{metrics['net_volume']:+,}"
    )

    st.markdown("---")

    # Row 3: Size metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Avg Size", f"{metrics['avg_trade_size']:.1f}")
    col2.metric("Max Size", f"{metrics['max_trade_size']:,}")
    col3.metric("Price Range", f"${metrics['price_range']:.2f}")
    col4.metric("Institutional Trades", f"{metrics['institutional_trades']:,}")


# =====================================================
# MAIN APPLICATION
# =====================================================

def main():
    """Main Streamlit application"""

    # Page config
    st.set_page_config(
        page_title="SPXW Option Explorer",
        page_icon="📊",
        layout="wide"
    )

    # Title
    st.title("SPXW Option Explorer")
    st.markdown("Interactive analysis tool for individual SPXW option contracts from Time & Sales data")
    st.markdown("---")

    # Render sidebar and get inputs
    inputs = render_sidebar()

    # Load data
    with st.spinner("Loading Time & Sales data..."):
        tas_data = load_tas_data(inputs['selected_date'])

    # === DAILY OVERVIEW SECTION ===
    with st.expander("📊 Daily Session Overview - Top Strikes", expanded=False):
        st.markdown("### Most Active Strikes Across Entire Session")

        # === EXPIRY BREAKDOWN ===
        st.markdown("#### Volume Breakdown by Expiration Date")
        expiry_breakdown = analyze_expiry_breakdown(tas_data)

        # Display expiry breakdown
        exp_cols = st.columns(len(expiry_breakdown))
        for idx, (_, row) in enumerate(expiry_breakdown.iterrows()):
            with exp_cols[idx]:
                st.metric(
                    label=row['expiry_label'],
                    value=f"{int(row['total_volume']):,} contracts",
                    delta=f"{row['total_volume']/expiry_breakdown['total_volume'].sum()*100:.1f}% of total"
                )
                st.caption(f"Notional: ${row['total_notional']:,.0f}")
                st.caption(f"Trades: {int(row['trade_count']):,}")

        st.markdown("---")

        col1, col2 = st.columns([7, 5])

        with col1:
            # Add expiry filter
            expiry_filter_options = ['All Expiries'] + expiry_breakdown['expiry_label'].tolist()
            selected_expiry_filter = st.selectbox(
                "Filter Top Strikes by Expiry",
                options=expiry_filter_options,
                index=0,
                key='daily_expiry_filter'
            )

            st.markdown("#### Top 10 by Volume")

            # Get all top strikes
            all_top_strikes = analyze_top_strikes(tas_data, top_n=100)  # Get more for filtering

            # Filter by selected expiry if not "All"
            if selected_expiry_filter != 'All Expiries':
                # Extract expiry_str from the label
                selected_expiry_str = expiry_breakdown[
                    expiry_breakdown['expiry_label'] == selected_expiry_filter
                ]['expiry_str'].iloc[0]

                # Parse and filter
                all_top_strikes['parsed_exp'] = all_top_strikes['expiry_str']
                top_strikes_vol = all_top_strikes[
                    all_top_strikes['parsed_exp'] == selected_expiry_str
                ].head(10)
            else:
                top_strikes_vol = all_top_strikes.head(10)

            # Format for display
            display_df = top_strikes_vol[['strike', 'option_type', 'expiry_label', 'total_volume', 'total_notional', 'vwap', 'trade_count']].copy()
            display_df.columns = ['Strike', 'Type', 'Expiry', 'Volume', 'Notional ($)', 'VWAP ($)', 'Trades']

            # Format numbers
            display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{int(x):,}")
            display_df['Notional ($)'] = display_df['Notional ($)'].apply(lambda x: f"${x:,.0f}")
            display_df['VWAP ($)'] = display_df['VWAP ($)'].apply(lambda x: f"${x:.2f}")
            display_df['Trades'] = display_df['Trades'].apply(lambda x: f"{int(x):,}")

            st.dataframe(display_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### Call vs Put Flow Summary")
            call_put_summary = analyze_call_put_flow(tas_data)

            # Create summary metrics
            calls = call_put_summary[call_put_summary['option_type'] == 'CALL']
            puts = call_put_summary[call_put_summary['option_type'] == 'PUT']

            call_volume = calls['size'].sum()
            put_volume = puts['size'].sum()
            total_volume = call_volume + put_volume

            call_notional = calls['notional'].sum()
            put_notional = puts['notional'].sum()
            total_notional = call_notional + put_notional

            # Display metrics
            st.metric("Total Volume", f"{int(total_volume):,} contracts")
            st.metric("Total Notional", f"${total_notional:,.0f}")

            st.markdown("---")

            # Call/Put breakdown
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Calls**")
                st.write(f"Vol: {int(call_volume):,}")
                st.write(f"{call_volume/total_volume*100:.1f}%")
            with c2:
                st.markdown("**Puts**")
                st.write(f"Vol: {int(put_volume):,}")
                st.write(f"{put_volume/total_volume*100:.1f}%")

            # Put/Call ratio
            pc_ratio = put_volume / call_volume if call_volume > 0 else 0
            st.markdown(f"**Put/Call Ratio:** {pc_ratio:.2f}")

        # Add chart
        st.markdown("---")
        fig = plot_top_strikes_chart(top_strikes_vol)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # Get available expiries for this strike and option type
    available_expiries = get_available_expiries(
        tas_data,
        inputs['strike'],
        inputs['option_type']
    )

    # Show expiry selector in main area
    if len(available_expiries) == 0:
        st.error(f"No trades found for {inputs['option_type']} strike {inputs['strike']}")
        st.info("Try a different strike price or option type. This strike may not have traded on this date.")
        st.markdown("**Suggestions:**")
        st.markdown("- Try a strike closer to the SPX price on this date")
        st.markdown("- Try the opposite option type (Call vs Put)")
        st.markdown("- Select a different trading date")
        st.stop()

    # Format expiry options for display
    expiry_options = ['All'] + [
        f"{exp} ({datetime.strptime(f'20{exp}', '%Y%m%d').strftime('%Y-%m-%d')})"
        for exp in available_expiries
    ]

    st.markdown("### Expiry Selection")
    col1, col2 = st.columns([3, 9])
    with col1:
        selected_expiry_display = st.selectbox(
            "Select Expiry",
            options=expiry_options,
            index=0,
            help="Choose a specific expiry or 'All' to see all expiries for this strike"
        )

    # Extract expiry string (YYMMDD) from selection
    if selected_expiry_display == 'All':
        selected_expiry = 'All'
        display_label = f"{inputs['option_type']} {inputs['strike']} (All Expiries)"
    else:
        selected_expiry = selected_expiry_display.split(' ')[0]  # Extract YYMMDD part
        expiry_date = datetime.strptime(f'20{selected_expiry}', '%Y%m%d').strftime('%Y-%m-%d')
        display_label = f"{inputs['option_type']} {inputs['strike']} (Exp: {expiry_date})"

    with col2:
        st.info(f"📊 Analyzing: **{display_label}** | Found **{len(available_expiries)}** expiry date(s)")

    st.markdown("---")

    # Filter for selected strike, option type, and expiry
    trades = filter_trades_by_strike_and_type(
        tas_data,
        inputs['strike'],
        inputs['option_type'],
        selected_expiry
    )

    # Check if data exists
    if len(trades) == 0:
        st.error(f"No trades found for the selected filters")
        st.info("Try selecting 'All' expiries or adjusting your filters.")
        st.stop()

    # Apply additional filters
    if inputs['min_size'] > 0 or inputs['time_range']:
        trades = apply_additional_filters(
            trades,
            inputs['min_size'],
            inputs['time_range']
        )

        if len(trades) == 0:
            st.error("No trades match the selected filters.")
            st.info("Try relaxing your filter criteria.")
            st.stop()

    # Update sidebar status
    st.sidebar.markdown("---")
    st.sidebar.success("Data Loaded")
    st.sidebar.metric("Trades Found", f"{len(trades):,}")

    # Calculate metrics
    metrics = calculate_flow_metrics(trades, inputs['institutional_threshold'])
    time_series = calculate_time_series_data(trades, window_minutes=5)
    institutional = identify_institutional_trades(trades, inputs['institutional_threshold'])

    # === TABS ===
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "💹 Buy/Sell Flow",
        "📈 Price & Volume",
        "📦 Trade Size Analytics",
        "📋 Data Table",
        "🔗 Spread Analysis",
        "⏰ Institutional Timing"
    ])

    with tab1:
        render_metrics_overview(metrics, display_label)

        # Summary table
        with st.expander("📊 Detailed Metrics Table"):
            metrics_df = pd.DataFrame([metrics]).T
            metrics_df.columns = ['Value']
            st.dataframe(metrics_df, use_container_width=True)

    with tab2:
        st.subheader("💹 Buy/Sell Flow Analysis")

        col1, col2 = st.columns(2)
        with col1:
            fig = plot_flow_breakdown(metrics)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            # Flow stats table
            flow_stats = pd.DataFrame({
                'Metric': ['BUY Volume', 'SELL Volume', 'Net Flow',
                          'BUY Notional', 'SELL Notional', 'Net Notional'],
                'Value': [
                    f"{metrics['buy_volume']:,}",
                    f"{metrics['sell_volume']:,}",
                    f"{metrics['net_volume']:+,}",
                    f"${metrics['buy_notional']:,.0f}",
                    f"${metrics['sell_notional']:,.0f}",
                    f"${metrics['net_notional']:+,.0f}"
                ]
            })
            st.table(flow_stats)

        st.markdown("---")
        fig = plot_time_series_flow(time_series)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        fig = plot_net_flow(time_series)
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.subheader("📈 Price & Volume Analysis")

        fig = plot_price_time_series(trades, time_series)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        fig = plot_cumulative_volume(time_series)
        st.pyplot(fig)
        plt.close(fig)

    with tab4:
        st.subheader("📦 Trade Size Analytics")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Institutional Trades", f"{len(institutional):,}")
            if metrics['total_volume'] > 0:
                inst_pct = institutional['size'].sum() / metrics['total_volume'] * 100
                st.metric("Institutional % of Volume", f"{inst_pct:.1f}%")
        with col2:
            if len(institutional) > 0:
                st.metric("Avg Institutional Size", f"{institutional['size'].mean():.1f}")
                st.metric("Max Institutional Size", f"{institutional['size'].max():,}")

        st.markdown("---")
        fig = plot_size_distribution(trades)
        st.pyplot(fig)
        plt.close(fig)

        if len(institutional) > 0:
            st.markdown("---")
            fig = plot_institutional_timeline(institutional, inputs['institutional_threshold'])
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(f"No institutional-sized trades (≥{inputs['institutional_threshold']} contracts) found.")

    with tab5:
        st.subheader("📋 All Trades")

        # Column selection
        default_cols = ['datetime', 'expiry_label', 'price', 'size', 'aggressor_side',
                       'notional', 'bid_price', 'ask_price', 'spx_price']
        available_cols = [c for c in default_cols if c in trades.columns]

        display_cols = st.multiselect(
            "Select columns to display",
            options=trades.columns.tolist(),
            default=available_cols
        )

        if display_cols:
            # Display dataframe
            st.dataframe(
                trades[display_cols],
                use_container_width=True,
                height=500
            )

            # Export options
            col1, col2 = st.columns(2)

            with col1:
                csv = trades.to_csv(index=False)
                # Create filename from inputs
                filename = f"option_trades_{inputs['option_type']}_{inputs['strike']}_{inputs['selected_date']}.csv"
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )

            with col2:
                st.info(f"Total rows: {len(trades):,}")
        else:
            st.warning("Please select at least one column to display.")

    with tab6:
        st.subheader("🔗 Spread Analysis")
        st.info("""
        **Note**: Spread detection analyzes the **full day's TAS data across ALL strikes and ALL expiries** to find multi-leg spreads.

        **Detects:**
        - **Traditional spreads** (opposite sides: BUY one strike, SELL another)
        - **Same-side spreads** (both BUY or both SELL - like short strangles, credit spreads)
        - **Vertical spreads** (same expiry, different strikes)
        - **Calendar spreads** (different expiry, same strike)

        Check the **Direction** column: OPPOSITE = traditional, SELL = both legs sold, BUY = both legs bought
        """)

        with st.spinner("Detecting spreads across all strikes..."):
            # Detect spreads - use FULL tas_data (all strikes), not filtered trades
            spreads = detect_spreads(tas_data, time_window_seconds=5)

        if len(spreads) == 0:
            st.warning("No spreads detected in the selected data.")
            st.info("""
            **Why no spreads detected?**
            - Spreads require simultaneous multi-leg execution within 5 seconds
            - Only institutional-sized legs (≥10 contracts) are analyzed for performance
            - This specific strike may not have had spread activity

            **Try:**
            - Select a different strike with higher volume
            - Remove time/size filters to see full day data
            - Check strikes near the money (higher spread activity)
            """)
        else:
            # Calculate metrics
            spread_metrics = calculate_spread_metrics(spreads)

            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Spreads", f"{spread_metrics['total_spreads']:,}")
            col2.metric("Total Notional", f"${spread_metrics['total_notional']:,.0f}")
            col3.metric("Avg Spread Width", f"${spread_metrics['avg_spread_width']:.2f}")
            col4.metric("Net Debit/Credit",
                       f"${spread_metrics['total_debit'] - spread_metrics['total_credit']:+,.0f}")

            st.markdown("---")

            # Spread type breakdown
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Spread Type Distribution**")
                fig = plot_spread_type_breakdown(spreads)
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                st.markdown("**Spread Types Summary**")
                type_summary = pd.DataFrame([
                    {'Type': k.capitalize(), 'Count': v}
                    for k, v in spread_metrics['by_type'].items()
                ])
                st.dataframe(type_summary, use_container_width=True, hide_index=True)

                st.markdown(f"**Total Debit:** ${spread_metrics['total_debit']:,.2f}")
                st.markdown(f"**Total Credit:** ${spread_metrics['total_credit']:,.2f}")

            st.markdown("---")

            # Hourly spread count
            st.markdown("**Spread Activity by Hour**")
            fig = plot_spread_count_by_hour(spreads)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("---")

            # Cumulative spreads
            st.markdown("**Cumulative Spread Detection**")
            fig = plot_cumulative_spreads(spreads)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("---")

            # Detailed spread table
            st.markdown("**Top 20 Detected Spreads (by notional)**")

            display_spreads = spreads.sort_values('total_notional', ascending=False).head(20).copy()

            # Format for display
            display_spreads['strikes_str'] = display_spreads['strikes'].apply(
                lambda x: ', '.join([str(int(s)) for s in x]) if isinstance(x, list) else str(x)
            )
            display_spreads['expiry_str'] = display_spreads['expiries'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
            )
            display_spreads['timestamp_str'] = pd.to_datetime(display_spreads['timestamp']).dt.strftime('%H:%M:%S')

            display_cols = ['timestamp_str', 'spread_type', 'spread_direction', 'option_type', 'expiry_str', 'strikes_str',
                           'total_size', 'net_debit_credit', 'total_notional', 'leg_count']

            rename_map = {
                'timestamp_str': 'Time',
                'spread_type': 'Type',
                'spread_direction': 'Direction',
                'option_type': 'Option',
                'expiry_str': 'Expiry',
                'strikes_str': 'Strikes',
                'total_size': 'Size',
                'net_debit_credit': 'Net Debit/Credit',
                'total_notional': 'Notional',
                'leg_count': 'Legs'
            }

            st.dataframe(
                display_spreads[display_cols].rename(columns=rename_map),
                use_container_width=True,
                hide_index=True
            )

            # Export button
            csv = spreads.to_csv(index=False)
            st.download_button(
                label="📥 Download All Spreads CSV",
                data=csv,
                file_name=f"spreads_{inputs['selected_date']}.csv",
                mime="text/csv"
            )

    with tab7:
        st.subheader("⏰ Institutional Timing Analysis")

        st.info("""
        **Note**: Timing analysis examines the **full day's TAS data** across all strikes to identify institutional flow patterns.
        The strike you selected in the sidebar is used only for filtering Tabs 1-5.
        """)

        st.markdown(f"""
        Analyzing temporal patterns in institutional-sized trades (≥{inputs['institutional_threshold']} contracts).
        This helps identify when smart money is most active.
        """)

        # Check if institutional trades exist - use FULL tas_data (all strikes)
        institutional = identify_institutional_trades(tas_data, inputs['institutional_threshold'])

        if len(institutional) == 0:
            st.warning(f"No institutional-sized trades (≥{inputs['institutional_threshold']} contracts) found.")
            st.info("""
            **No institutional activity detected for this strike.**

            This could mean:
            - This strike had mostly retail-sized trades
            - The strike is too far OTM/ITM for institutional interest
            - Try a strike closer to the money
            """)
        else:
            # Summary metrics
            inst_metrics = calculate_flow_metrics(institutional, inputs['institutional_threshold'])

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Institutional Trades", f"{len(institutional):,}")
            col2.metric("Total Volume", f"{inst_metrics['total_volume']:,}")
            col3.metric("Avg Trade Size", f"{inst_metrics['avg_trade_size']:.1f}")
            col4.metric("Net Flow", f"{inst_metrics['net_volume']:+,}")

            st.markdown("---")

            # Hourly analysis
            st.markdown("**Hourly Institutional Volume**")
            hourly = analyze_hourly_flow(trades, inputs['institutional_threshold'])

            if len(hourly) > 0:
                fig = plot_hourly_institutional_volume(hourly, inputs['institutional_threshold'])
                st.pyplot(fig)
                plt.close(fig)

                with st.expander("📊 View Hourly Data Table"):
                    st.dataframe(hourly, use_container_width=True, hide_index=True)
            else:
                st.info("No hourly data available")

            st.markdown("---")

            # 30-minute windows
            st.markdown("**30-Minute Window Analysis**")
            windows = analyze_30min_windows(trades, inputs['institutional_threshold'])

            if len(windows) > 0:
                fig = plot_30min_heatmap(windows)
                st.pyplot(fig)
                plt.close(fig)

                with st.expander("📊 View 30-Minute Window Data"):
                    st.dataframe(windows, use_container_width=True, hide_index=True)
            else:
                st.info("No 30-minute window data available")

            st.markdown("---")

            # Market period comparison
            st.markdown("**Market Period Comparison**")
            period_results = analyze_market_periods(trades, inputs['institutional_threshold'])

            fig = plot_market_period_comparison(period_results)
            st.pyplot(fig)
            plt.close(fig)

            # Period summary table
            with st.expander("📊 View Period Details"):
                period_df = pd.DataFrame(period_results).T
                period_df.index = ['Opening (9:30-10:30)', 'Midday (12:00-14:00)', 'Power Hour (15:00-16:00)']
                st.dataframe(period_df, use_container_width=True)

            st.markdown("---")

            # Cumulative volume over time
            st.markdown("**Cumulative Institutional Volume**")
            fig = plot_institutional_cumulative_volume(trades, inputs['institutional_threshold'])
            st.pyplot(fig)
            plt.close(fig)

            # Key insights
            st.markdown("---")
            st.markdown("**📈 Key Insights**")

            # Find most active hour
            if len(hourly) > 0:
                most_active_hour = hourly.loc[hourly['trade_count'].idxmax()]
                st.info(f"Most active hour: **{most_active_hour['hour_label']}** with {int(most_active_hour['trade_count'])} trades")

            # Find dominant period
            period_volumes = {k: v['trade_count'] for k, v in period_results.items()}
            dominant_period = max(period_volumes, key=period_volumes.get)
            period_names = {
                'opening': 'Opening (9:30-10:30)',
                'midday': 'Midday (12:00-14:00)',
                'power_hour': 'Power Hour (15:00-16:00)'
            }
            st.info(f"Most active period: **{period_names[dominant_period]}** with {period_volumes[dominant_period]} trades")

            # Net bias
            if inst_metrics['buy_pct'] > 60:
                st.success(f"Strong **BUY** bias: {inst_metrics['buy_pct']:.1f}% buy-side aggression")
            elif inst_metrics['buy_pct'] < 40:
                st.error(f"Strong **SELL** bias: {100-inst_metrics['buy_pct']:.1f}% sell-side aggression")
            else:
                st.warning(f"Balanced flow: {inst_metrics['buy_pct']:.1f}% buy / {100-inst_metrics['buy_pct']:.1f}% sell")


if __name__ == "__main__":
    main()
