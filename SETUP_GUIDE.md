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




