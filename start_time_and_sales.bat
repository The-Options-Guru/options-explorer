@echo off
setlocal enabledelayedexpansion
REM SPXW Time and Sales Collector - Simple Auto-Start
REM Double-click anytime - waits until start time, then launches collector
REM Press Ctrl+C in the Python window to stop when done

REM ========================================
REM EDIT THIS VALUE FOR YOUR TIMEZONE
REM ========================================
REM Start collecting at this time (24-hour format)
set /a START_HOUR=9
set /a START_MINUTE=45
REM ========================================

echo ========================================
echo SPXW and SPY Time and Sales Collectors
echo Auto-start: %START_HOUR%:%START_MINUTE%
echo ========================================
echo.

:check_time
REM Get current time in 24-hour format
for /f "tokens=1-2 delims=:." %%a in ("%time%") do (
    set HOUR=%%a
    set MINUTE=%%b
)

REM Remove leading spaces/zeros to avoid octal interpretation
set HOUR=!HOUR: =!
set MINUTE=!MINUTE: =!

REM Strip leading zeros (08 and 09 are invalid octal)
if "!HOUR:~0,1!"=="0" set HOUR=!HOUR:~1!
if "!MINUTE:~0,1!"=="0" set MINUTE=!MINUTE:~1!

REM Handle empty strings
if "!HOUR!"=="" set HOUR=0
if "!MINUTE!"=="" set MINUTE=0

REM Now safe to do arithmetic
set /a HOUR=!HOUR!
set /a MINUTE=!MINUTE!

echo Current time: !HOUR!:!MINUTE!

REM Calculate total minutes for easier comparison
set /a CURRENT_TOTAL=!HOUR!*60+!MINUTE!
set /a START_TOTAL=%START_HOUR%*60+%START_MINUTE%

REM Check if before start time
if !CURRENT_TOTAL! LSS !START_TOTAL! (
    echo Waiting until %START_HOUR%:%START_MINUTE% to start...
    timeout /t 30 /nobreak > nul
    goto check_time
)

REM Start collecting
echo.
echo Starting collectors at !HOUR!:!MINUTE!
echo Press Ctrl+C in each collector window to stop when done
echo.

REM Start SPXW collector in new window
start "SPXW Collector" python spxw_time_and_sales.py

REM Start SPY collector in new window
start "SPY Collector" python spy_time_and_sales.py

echo.
echo Both collectors started in separate windows.
pause
