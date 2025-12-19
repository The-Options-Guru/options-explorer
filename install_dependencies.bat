@echo off
echo ========================================
echo   Options Flow Explorer - Installation
echo ========================================
echo.
echo Installing required Python packages...
echo This may take a few minutes.
echo.
echo Packages being installed:
echo   - streamlit (web app)
echo   - pandas, numpy (data processing)
echo   - matplotlib, seaborn (charts)
echo   - websocket-client (data collector)
echo.
echo ========================================
echo.

python -m pip install -r requirements.txt

echo.
echo ========================================
echo.
echo Installation complete!
echo.
echo You can now:
echo   - Run start_explorer.bat to launch the Option Explorer
echo   - Run time_and_sales_collector.py to collect data
echo   - Run Start collector.bat for scheduled data collection
echo.
echo ========================================
echo.

pause
