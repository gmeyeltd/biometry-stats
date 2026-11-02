@echo off
REM Runs the whole WHWK vs Moussa shared-index study. Double-click, or run from an Anaconda Prompt.
cd /d "%~dp0"

REM ---- find Python (active conda env first, then usual Anaconda/Miniconda folders) ----
set "PY="
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PY=%CONDA_PREFIX%\python.exe"
if not defined PY for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "C:\ProgramData\anaconda3" "C:\ProgramData\miniconda3" "%USERPROFILE%\Anaconda3" "%USERPROFILE%\Miniconda3") do (
  if not defined PY if exist "%%~P\python.exe" set "PY=%%~P\python.exe"
)
if not defined PY (
  echo Could not find conda Python. Open "Anaconda Prompt", cd to this folder and run run_all.bat from there.
  goto :err
)
REM conda's DLLs (numpy etc.) need these on PATH when not launched from an Anaconda Prompt
for %%D in ("%PY%") do set "PYDIR=%%~dpD"
set "PATH=%PYDIR%;%PYDIR%Library\mingw-w64\bin;%PYDIR%Library\usr\bin;%PYDIR%Library\bin;%PYDIR%Scripts;%PATH%"
echo Using Python: %PY%

echo === Step 0: prepare data ===
"%PY%" 00_prepare_data.py || goto :err

echo === Step 1: WHWK reference (R) ===
set "RSCRIPT="
where Rscript >nul 2>nul && set "RSCRIPT=Rscript"
if not defined RSCRIPT for /d %%D in ("C:\Program Files\R\R-*") do set "RSCRIPT=%%D\bin\Rscript.exe"
if not defined RSCRIPT (
  echo Could not find R. Edit RSCRIPT in this file to point at Rscript.exe.
  goto :err
)
echo Using R: %RSCRIPT%
"%RSCRIPT%" 01_whwk_reference.R || goto :err

echo === Step 2: Moussa (Python) on shared indices + comparison ===
"%PY%" 02_moussa_and_compare.py || goto :err

echo.
echo Done. Results are in the outputs folder.
pause
exit /b 0

:err
echo.
echo *** A step failed - copy the messages above to Claude. ***
pause
exit /b 1
