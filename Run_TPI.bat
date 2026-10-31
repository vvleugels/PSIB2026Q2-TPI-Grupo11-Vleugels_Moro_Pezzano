@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Setting up the TPI environment. This may take a few minutes...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo.
        echo Could not create the environment. Install 64-bit Python 3.12 and its py launcher.
        pause
        exit /b 1
    )
)

echo Checking and installing required packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency installation failed. Check your internet connection and try again.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 (
    echo.
    echo The environment has incompatible dependencies. Resolve the errors above and try again.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\run_tpi.py
if errorlevel 1 (
    echo.
    echo TPI failed. See the error above.
    pause
    exit /b 1
)

pause
endlocal
