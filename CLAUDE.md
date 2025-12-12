# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPXW Option Explorer is an interactive Streamlit application for analyzing individual SPXW option contracts using Time & Sales (TAS) data. The application provides comprehensive flow analysis, institutional trade detection, spread pattern recognition, and temporal analytics for options trading.

## Core Data Format

The application expects CSV files with the naming pattern `spxw_tas_data_YYYY-MM-DD.csv` in the root directory. Required columns:
- `date`, `time`, `symbol`, `price`, `size`, `aggressor_side`
- Optional: `bid_price`, `ask_price`, `spx_price`, `spx_bid`, `spx_ask`, `sequence`

## Running the Application

```bash
# Run the Streamlit application
streamlit run option_explorer.py

# Specify port
streamlit run option_explorer.py --server.port 8501
```

## Key Architecture Components

### Option Symbol Format
SPXW option symbols follow the pattern: `.SPXW[YYMMDD][C/P][STRIKE]`
- Example: `.SPXW251211C6775` represents a CALL option expiring Dec 11, 2025 at strike 6775
- The `parse_option_symbol()` and `construct_option_symbol()` functions handle conversion

### Data Processing Pipeline
1. **Data Loading** (`load_tas_data`): Loads CSV, validates columns, creates datetime index, calculates notional values
2. **Filtering** (`filter_trades_by_symbol`, `apply_additional_filters`): Isolates specific option contracts and applies size/time filters
3. **Analytics** (`calculate_flow_metrics`, `calculate_time_series_data`): Aggregates buy/sell metrics, VWAP, institutional trades

### Major Feature Modules

**Flow Analytics** (lines 172-224)
- `calculate_flow_metrics()`: Computes buy/sell volume, notional, VWAP, institutional trade counts
- Classifies trades by aggressor side for directional flow analysis

**Time Series Aggregation** (lines 227-273)
- `calculate_time_series_data()`: Aggregates trades into N-minute windows
- Computes cumulative buy/sell, VWAP per window, net flow over time

**Spread Detection** (lines 285-505)
- `detect_spreads()`: Identifies vertical and calendar spreads from trade sequences
- Uses time proximity (5 seconds) and size matching (1:1, 2:1 ratios)
- Only analyzes institutional-sized legs (≥10 contracts) for performance
- Returns structured spread data with leg details, net debit/credit

**Institutional Timing Analysis** (lines 512-630)
- `analyze_hourly_flow()`: Aggregates institutional trades by hour
- `analyze_30min_windows()`: 30-minute window breakdowns
- `analyze_market_periods()`: Opening, midday, power hour comparisons

### Visualization System

All charts use Matplotlib with seaborn styling (`seaborn-v0_8-darkgrid`). Key chart functions:
- `plot_flow_breakdown()`: Pie chart for buy/sell distribution
- `plot_time_series_flow()`: Stacked area chart with aggressor-side volume
- `plot_price_time_series()`: Scatter plot with VWAP overlay (bubble size = trade size)
- `plot_cumulative_volume()`: Cumulative buy/sell throughout day
- `plot_institutional_timeline()`: Large trade scatter plot
- `plot_spread_type_breakdown()`: Spread pattern distribution
- `plot_hourly_institutional_volume()`: Institutional flow by hour
- `plot_30min_heatmap()`: Heatmap of 30-minute windows

### UI Structure (Tabs)

1. **Overview**: High-level metrics and summary statistics
2. **Buy/Sell Flow**: Directional flow charts and net flow analysis
3. **Price & Volume**: Price action with VWAP, cumulative volume
4. **Trade Size Analytics**: Size distribution, institutional trade timeline
5. **Data Table**: Raw trade data with column selection and CSV export
6. **Spread Analysis**: Multi-leg spread detection across all strikes (full day)
7. **Institutional Timing**: Temporal patterns for large trades (full day)

**Important**: Tabs 6 and 7 analyze the FULL day's TAS data across all strikes, not just the selected strike. This is intentional for detecting spreads and timing patterns.

## Key Constants and Thresholds

- `OPTION_MULTIPLIER = 100`: Standard option contract multiplier
- `INSTITUTIONAL_SIZE_THRESHOLD = 20`: Default minimum contracts for institutional classification (adjustable in UI)
- Spread detection time window: 5 seconds
- Spread detection sequence proximity: 50 sequence numbers
- Time series default aggregation: 5-minute windows

## Data Caching

The application uses Streamlit's `@st.cache_data` decorator on:
- `get_available_dates()`: Scans directory for available CSV files
- `load_tas_data(date)`: Loads and preprocesses CSV data

Clear cache if data files are modified: press 'C' in Streamlit or restart the app.

## Common Development Tasks

### Adding New Analytics
1. Create analytics function in the appropriate section (lines 170-630)
2. Add corresponding chart function (lines 635-1052)
3. Update the relevant tab in `main()` (lines 1256-1622)

### Modifying Spread Detection
- Adjust `time_window_seconds` parameter in `detect_spreads()` (line 390)
- Modify size ratio thresholds in `identify_2leg_spread()` (line 327)
- Add new spread patterns by extending `identify_2leg_spread()` logic

### Adding New Chart Types
- Follow existing pattern: create function returning `fig` object
- Use `plt.close(fig)` after `st.pyplot(fig)` to prevent memory leaks
- Apply consistent styling with existing charts

## Performance Considerations

- Spread detection filters to institutional size (≥10 contracts) for performance
- Time bucketing used in spread detection to limit pairwise comparisons
- Sequence proximity check limits lookahead to 50 trades
- Streamlit caching minimizes repeated CSV reads

## Dependencies

Core libraries:
- `streamlit`: Web application framework
- `pandas`, `numpy`: Data processing
- `matplotlib`, `seaborn`: Visualization
- Standard library: `datetime`, `pathlib`, `glob`, `re`
