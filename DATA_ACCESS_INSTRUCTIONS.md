# Data Access Instructions - For End Users

**Welcome to SPXW Option Explorer!**

The app is free, but requires SPXW Time & Sales data files to function. Data access is provided separately.

---

## Getting Data Access

### Step 1: Request Access

Contact the data provider to request access:
- **https://discord.gg/r7a6vawX8c**

### Step 2: Receive Access

Once approved, you'll receive:
- Link to shared data folder (Google Drive, Dropbox, etc.)
- Access credentials (if needed)

### Step 3: Download Data

1. Open the shared data folder link
2. Download the CSV file(s) you need
3. Files are named: `spxw_tas_data_YYYY-MM-DD.csv`

---

## Setting Up Data Files

### Where to Put Data Files:

**Place CSV files in the same folder as `option_explorer.py`**

Example folder structure:
```
SPXW_Option_Explorer/
├── option_explorer.py
├── start_explorer.bat
├── spxw_tas_data_2025-12-11.csv  ← Your data files go here
├── spxw_tas_data_2025-12-12.csv
└── spxw_tas_data_2025-12-13.csv
```

### File Naming Important:

Files MUST be named exactly: `spxw_tas_data_YYYY-MM-DD.csv`

✅ Correct:
- `spxw_tas_data_2025-12-12.csv`
- `spxw_tas_data_2025-01-05.csv`

❌ Wrong:
- `data.csv`
- `spxw_12-12-2025.csv`
- `spxw_tas_data_12-12-25.csv`

---

## Daily Updates

### Getting New Data:

1. **Check for new files** in the shared folder (usually available after market close)
2. **Download latest CSV** file
3. **Place in app folder** alongside existing files
4. **Restart app** to see new date in dropdown

### Automatic Refresh:

- The app will automatically detect new files
- Just add them to the folder and they'll appear in the date selector

---

## Data File Requirements

Your CSV files must have these columns:

**Required:**
- `date` - Trading date
- `time` - Trade timestamp
- `symbol` - Option symbol
- `price` - Trade price
- `size` - Contract size
- `aggressor_side` - "BUY" or "SELL"

**Optional (but recommended):**
- `bid_price` - Option bid at trade time
- `ask_price` - Option ask at trade time
- `spx_price` - SPX price at trade time
- `sequence` - Trade sequence number

---

## Troubleshooting

### "No TAS data files found!"

**Problem:** App can't find CSV files

**Solutions:**
1. Check files are in the correct folder (same as `option_explorer.py`)
2. Verify filename format: `spxw_tas_data_YYYY-MM-DD.csv`
3. Make sure files aren't still zipped (extract if needed)
4. Check file extensions are `.csv` not `.csv.txt`

### "No trades found for this strike"

**Problem:** Selected strike didn't trade on that date

**Solutions:**
1. Try a different strike price (closer to SPX price)
2. Try the opposite option type (Call vs Put)
3. Select "All" expiries to see all available data
4. Check the Daily Overview to see which strikes were most active

### "Missing columns in data file"

**Problem:** CSV is missing required columns

**Solutions:**
1. Contact data provider - file may be corrupted
2. Re-download the file
3. Verify you're using the correct data file

### Files are very large / slow to load

**Solutions:**
1. Download only the dates you need
2. Delete old files you no longer use
3. If files are zipped, keep them zipped until needed

---

## Data Storage Tips

### How Many Files to Keep:

**Option A: Keep last 7-30 days**
- Reasonable storage usage
- Good for recent analysis
- Delete older files

**Option B: Keep all historical data**
- Better for long-term analysis
- More storage required
- Organize by month

### Storage Space:

Typical file sizes:
- 1 day of data: 50-400MB (varies by market activity)
- 30 days: 1.5-12GB
- 1 year: 18-150GB

---

## Data Privacy & Terms

### Important Notes:

1. **Do not redistribute data files** - Your access is personal
2. **Do not share access credentials** - Keep your access secure
3. **Data is for analysis only** - Check terms of use for restrictions
4. **Access can be revoked** - Follow the terms to maintain access

### Terms of Use:

By downloading and using the data, you agree to:
- Use data for personal analysis only
- Not redistribute or share data files
- Follow any additional terms provided by data provider

---

## Getting Help

### For App Issues:
- Check README.md
- Check SETUP_GUIDE.md
- Check GitHub issues (if app is on GitHub)

### For Data Access Issues:
- Contact the data provider
- Check your access hasn't expired
- Verify you're using the correct credentials

---

## FAQ

**Q: Is the data free?**
A: The app is free. Data access is provided separately and may be free or paid depending on the provider's policy.

**Q: How often is new data available?**
A: New data is typically uploaded daily after market close (around 4:30-5:00 PM ET).

**Q: Can I analyze old data?**
A: Yes! Any CSV file you have access to can be loaded and analyzed.

**Q: What if I lose access?**
A: Contact the data provider to renew or restore access.

**Q: Can I use my own data?**
A: Yes! As long as it's in the correct CSV format with required columns.

**Q: How long does data access last?**
A: Check with your data provider for terms of your access.

**Q: Can I share data with a colleague?**
A: No - each user must request their own access. Sharing violates terms of use.

---

## Quick Start Checklist

Once you have access:

□ Download data files from shared folder
□ Place files in app folder (same location as option_explorer.py)
□ Verify filenames match: `spxw_tas_data_YYYY-MM-DD.csv`
□ Run start_explorer.bat
□ Select date from dropdown
□ Start analyzing!

---

**Ready to explore? Fire up the app and dive into the data!** 🚀


