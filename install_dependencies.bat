@echo off
echo ========================================
echo   SPXW Option Explorer - Installation
echo ========================================
echo.
echo Installing required Python packages...
echo This may take a few minutes.
echo.
echo ========================================
echo.

python -m pip install streamlit pandas numpy matplotlib seaborn

echo.
echo ========================================
echo.
echo Installation complete!
echo.
echo You can now run start_explorer.bat to launch the app.
echo.
echo ========================================
echo.

pause
