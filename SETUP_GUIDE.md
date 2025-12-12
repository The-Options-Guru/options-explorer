# SPXW Option Explorer - Setup Guide

This guide will help you set up the SPXW Option Explorer, even if you don't have Python installed.

## Option 1: Fresh Install (No Python Installed)

### Step 1: Install Python

1. **Download Python:**
   - Go to: https://www.python.org/downloads/
   - Download Python 3.9 or newer (3.11 recommended)

2. **Install Python:**
   - Run the installer
   - **CRITICAL:** Check the box "Add Python to PATH" during installation
   - Click "Install Now"

3. **Verify Installation:**
   - Open Command Prompt (search "cmd" in Windows)
   - Type: `python --version`
   - Should show: `Python 3.x.x`

### Step 2: Install the App

1. **Extract the app folder** to a location like `C:\SPXW_Explorer\`

2. **Install dependencies:**
   - Double-click `install_dependencies.bat`
   - Wait for it to complete (takes 2-5 minutes)

3. **Run the app:**
   - Double-click `start_explorer.bat`
   - App will open in your browser at http://localhost:8501

---

## Option 2: Already Have Python

### Quick Install:

1. Open Command Prompt in the app folder
2. Run: `pip install -r requirements.txt`
3. Run: `streamlit run option_explorer.py`

---

## What Files to Share

When sharing this app, include these files:

### Required Files:
```
option_explorer.py          # Main application
requirements.txt            # Python dependencies
README.md                   # User guide
SETUP_GUIDE.md             # start_explorer.bat          # Windows startup script
install_dependencies.bat    # Windows install script
```

### Data Files:
```
spxw_tas_data_YYYY-MM-DD.csv  # Your TAS data files
```

**Note:** Data files can be large. Consider:
- Sharing sample data only
- Using file sharing services (Google Drive, Dropbox, etc.)
- Compressing data files (.zip)

---

## Data Requirements

### CSV Format Required:
Your CSV files must have these columns:
- `date`, `time`, `symbol`, `price`, `size`, `aggressor_side`

Optional but recommended:
- `bid_price`, `ask_price`, `spx_price`, `sequence`

### File Naming:
Files must be named: `spxw_tas_data_YYYY-MM-DD.csv`

Example: `spxw_tas_data_2025-12-11.csv`

---

## Troubleshooting

### "Python not found"
- Reinstall Python and check "Add to PATH"
- Restart your computer after installing

### "pip not recognized"
- Use: `python -m pip install -r requirements.txt`

### Port 8501 already in use
- Close other instances of the app
- Or edit `start_explorer.bat` and change port to 8502

### Data not loading
- Check CSV filename format matches exactly
- Ensure CSV has all required columns
- Check that data files are in same folder as `option_explorer.py`

---

## Alternative: Online Hosting (Advanced)

For users who can't install Python, consider hosting on Streamlit Cloud:

1. **Create GitHub repo** with your code
2. **Sign up:** https://streamlit.io/cloud
3. **Deploy:** Connect your GitHub repo
4. **Share:** Users access via URL (no install needed)

**Note:** Free tier has limitations on data size and usage.

---

## System Requirements

- **OS:** Windows 10/11, macOS, or Linux
- **RAM:** 4GB minimum, 8GB recommended
- **Disk:** 500MB for Python + dependencies
- **Data:** Additional space for CSV files (varies)

---

## Security Note

This app runs locally on your machine. Data stays on your computer and is not sent anywhere online.

---

## Support

If you encounter issues:
1. Check this guide first
2. Verify all required files are present
3. Ensure CSV data format is correct
4. Try restarting the app

