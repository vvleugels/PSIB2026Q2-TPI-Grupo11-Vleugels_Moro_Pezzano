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
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Dependency installation failed. Check your internet connection and try again.
        pause
        exit /b 1
    )
)

if not exist "data\Selected_database" (
    echo.
    echo The bundled data folder data\Selected_database was not found.
    echo Restore it or update data_dir in config.yaml.
    pause
    exit /b 1
)

echo Selecting 2D slices from the configured cases...
".venv\Scripts\python.exe" scripts\select_slices_2d.py
if errorlevel 1 (
    echo.
    echo Slice selection failed. See the error above.
    pause
    exit /b 1
)

set "RUN_ID=base-%RANDOM%"
echo Running the baseline segmentation experiment (%RUN_ID%)...
".venv\Scripts\python.exe" scripts\run_segmentation.py --run-id "%RUN_ID%"
if errorlevel 1 (
    echo.
    echo The segmentation experiment failed. See the error above.
    pause
    exit /b 1
)

echo.
echo TPI run completed. Results are in results\segmentation\%RUN_ID%.
pause
endlocal
