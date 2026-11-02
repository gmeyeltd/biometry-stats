@echo off
REM Runtime benchmark: WHWK (R) vs Moussa Research Suite (Python). Run from an Anaconda Prompt: .\run_benchmark.bat
REM Close other heavy programs first. R part takes about 60-75 min (datasets run one at a time), Python part about 5 min.
cd /d "%~dp0"

set "PY="
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PY=%CONDA_PREFIX%\python.exe"
if not defined PY for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "C:\ProgramData\anaconda3") do (
  if not defined PY if exist "%%~P\python.exe" set "PY=%%~P\python.exe"
)
if not defined PY ( echo Could not find conda Python. & goto :err )

set "RSCRIPT="
where Rscript >nul 2>nul && set "RSCRIPT=Rscript"
if not defined RSCRIPT for /d %%D in ("C:\Program Files\R\R-*") do set "RSCRIPT=%%D\bin\Rscript.exe"
if not defined RSCRIPT ( echo Could not find R. & goto :err )

echo === Benchmark 1/2: WHWK in R ===
"%RSCRIPT%" 03_benchmark.R || goto :err
echo === Benchmark 2/2: Moussa in Python ===
"%PY%" 03_benchmark.py || goto :err
echo.
echo Done. See outputs\benchmark_summary.csv
pause
exit /b 0

:err
echo *** A step failed - copy the messages above to Claude. ***
pause
exit /b 1
