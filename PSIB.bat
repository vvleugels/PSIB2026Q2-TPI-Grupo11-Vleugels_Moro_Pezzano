@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Setting up the TPI environment. This may take a few minutes...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Could not create the environment. Install 64-bit Python 3.12 and its py launcher.
        pause
        exit /b 1
    )
)

echo Checking installed packages...
".venv\Scripts\python.exe" scripts\check_requirements.py >nul 2>&1
set NEED_INSTALL=0
if errorlevel 1 set NEED_INSTALL=1
".venv\Scripts\python.exe" -m pip check >nul 2>&1
if errorlevel 1 set NEED_INSTALL=1

if "%NEED_INSTALL%"=="1" (
    echo Installing or repairing required packages...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed. Check your internet connection and try again.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" scripts\check_requirements.py
    if errorlevel 1 (
        echo One or more required packages are still missing or incompatible.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip check
    if errorlevel 1 (
        echo The environment still has incompatible dependencies. Resolve the errors above and try again.
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\pythonw.exe" (
    echo pythonw.exe was not found in the virtual environment.
    pause
    exit /b 1
)
start "TPI" ".venv\Scripts\pythonw.exe" scripts\gui.py
if errorlevel 1 (
    echo Could not start the TPI graphical interface.
    pause
    exit /b 1
)
endlocal
