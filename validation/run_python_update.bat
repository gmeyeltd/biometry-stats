@echo off
REM Re-runs the Python side only (R results and resamples from run_all.bat are reused).
REM Use after updating biometry_stats.py. Takes a few minutes. Run: .\run_python_update.bat
cd /d "%~dp0"
set "PY="
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PY=%CONDA_PREFIX%\python.exe"
if not defined PY for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "C:\ProgramData\anaconda3") do (
  if not defined PY if exist "%%~P\python.exe" set "PY=%%~P\python.exe"
)
if not defined PY ( echo Could not find conda Python. & goto :err )
if exist outputs\benchmark_threads.csv del outputs\benchmark_threads.csv
echo === 1/3 Exactness vs R (shared resamples) ===
"%PY%" 02_moussa_and_compare.py || goto :err
echo === 2/3 Deviations and own-seed runs ===
"%PY%" 04_deviations.py || goto :err
echo === 3/3 Python runtime (R timings reused) ===
"%PY%" 03_benchmark.py || goto :err
echo.
echo Done. Tell Claude.
pause
exit /b 0
:err
echo *** A step failed - copy the messages above to Claude. ***
pause
exit /b 1
