# SPXW Option Explorer

Interactive Streamlit application for analyzing SPXW option contracts from Time & Sales data.

**🆓 Free Application | 🔒 Controlled Data Access**

The app is free and open-source. SPXW Time & Sales data is provided separately under controlled access.

## Quick Start

### 1. Install Dependencies (One Time Only)

**Option A: Double-click the install file (Easiest)**

Just double-click `install_dependencies.bat` and wait for it to complete.

**Option B: Command line**

```bash
pip install streamlit pandas numpy matplotlib seaborn
```

### 2. Run the Application

**Option A: Double-click the batch file (Easiest)**

Just double-click `start_explorer.bat` - that's it!

**Option B: Command line**

```bash
streamlit run option_explorer.py
```

The app will automatically open in your browser at `http://localhost:8501`

### 3. Stop the Application

**To stop when you're done:**
- Press `Ctrl+C` in the command window, OR
- Simply close the command window
- The browser tab can stay open or be closed

### Alternative: Specify Port

```bash
streamlit run option_explorer.py --server.port 8080
```

## What Does This Do?

This tool analyzes individual SPXW option contracts and provides:

- **Buy/Sell Flow Analysis**: Track directional order flow and aggressor-side activity
- **Price & Volume Charts**: Visualize trade prices, VWAP, and volume patterns
- **Institutional Trade Detection**: Identify and analyze large block trades
- **Spread Detection**: Automatically detect vertical and calendar spreads
- **Timing Analysis**: See when institutional traders are most active (hourly, 30-min windows)
- **Raw Data Export**: Download filtered trade data as CSV

## Data Access

**The app requires SPXW Time & Sales data files to function.**

### Getting Data Access:

Data files are provided separately under controlled access. To request access:
- **See DATA_ACCESS_INSTRUCTIONS.md for details**

Once you have access, download CSV files and place them in the app folder.

### Data File Format:

Files must be named: `spxw_tas_data_YYYY-MM-DD.csv`

Example: `spxw_tas_data_2025-12-11.csv`

### Required CSV Columns

Your CSV files must have these columns:
- `date` - Trading date
- `time` - Trade timestamp
- `symbol` - Option symbol (e.g., `.SPXW251211C6775`)
- `price` - Trade price
- `size` - Contract size
- `aggressor_side` - Either "BUY" or "SELL"

### Optional Columns (Enhance Analysis)

- `bid_price`, `ask_price` - Option bid/ask at trade time
- `spx_price`, `spx_bid`, `spx_ask` - Underlying SPX prices
- `sequence` - Trade sequence number (improves spread detection)

## How to Use

1. **Start the app** using the command above
2. **Select a date** from the sidebar (shows all available CSV files)
3. **Enter strike price** (e.g., 6800)
4. **Choose CALL or PUT**
5. **Adjust filters** (optional):
   - Institutional size threshold (default: 20 contracts)
   - Minimum trade size filter
   - Time range filter
6. **Explore the tabs**:
   - Overview: Summary metrics
   - Buy/Sell Flow: Directional analysis
   - Price & Volume: Price action and VWAP
   - Trade Size Analytics: Size distribution and large trades
   - Data Table: Raw trade data with export
   - Spread Analysis: Multi-leg spread detection
   - Institutional Timing: When big players are active

## Troubleshooting

### "No TAS data files found!"

- Make sure your CSV files are in the same directory as `option_explorer.py`
- Check the filename format: `spxw_tas_data_2025-12-11.csv`

### "No trades found for this symbol"

- The strike price you entered may not have traded on that date
- Try a strike closer to the SPX price on that date
- Try switching between CALL and PUT

### "Missing columns in data file"

- Verify your CSV has all required columns listed above
- Column names are case-sensitive

### Port Already in Use

If port 8501 is busy:
```bash
streamlit run option_explorer.py --server.port 8502
```

### Clear Cache

If data looks stale after updating CSV files, press **C** in the Streamlit app to clear cache and reload.

## Example Workflow

```bash
# 1. Navigate to the directory
cd C:\Users\uer#1\Desktop\opt_exp

# 2. Make sure you have your CSV files
# spxw_tas_data_2025-12-11.csv
# spxw_tas_data_2025-12-10.csv
# etc.

# 3. Run the app
streamlit run option_explorer.py

# 4. App opens in browser automatically
# If not, go to: http://localhost:8501
```

## Tips

- **Spread Analysis** and **Institutional Timing** tabs analyze the entire day across all strikes, not just your selected strike
- Use the **institutional threshold slider** to adjust what qualifies as a "large" trade
- **Download CSV** button in Data Table tab lets you export filtered trades
- Charts show bubble sizes proportional to trade size
- Green = BUY aggressor, Red = SELL aggressor

## Need Help?

- Check that your CSV files match the expected format
- Verify all required columns are present
- Try selecting a different strike price or date
- Use the filters to narrow down your analysis

---

**Made for analyzing SPXW option flow from Time & Sales data**

