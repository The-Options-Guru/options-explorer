@echo off
REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo ========================================
echo   SPXW Option Explorer
echo ========================================
echo.

REM Check if streamlit is installed
python -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Streamlit is not installed!
    echo.
    echo FIRST TIME SETUP:
    echo   1. Double-click install_dependencies.bat
    echo   2. Wait for installation to complete
    echo   3. Then run this file again
    echo.
    echo ========================================
    pause
    exit /b 1
)

echo Starting Streamlit application...
echo.
echo IMPORTANT: After you press Enter at the email prompt,
echo wait a few seconds, then manually go to:
echo.
echo     http://localhost:8501
echo.
echo in your web browser.
echo.
echo To STOP the app:
echo   - Press Ctrl+C in this window, OR
echo   - Simply close this window
echo.
echo ========================================
echo.

python -m streamlit run option_explorer.py --server.headless=false

pause
