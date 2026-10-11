@echo off
rem ThermoSim launcher for Windows: double-click to open the GUI.
rem
rem Works in two situations:
rem   1. This file sits in a clone of the repository (packaging\windows\thermosim.bat).
rem      It runs the clone with "uv run", which installs the dependencies the first time.
rem   2. ThermoSim was installed with "uv tool install". It then runs the installed program.
rem
rem Arguments are passed on, so "thermosim.bat my_setup.toml" opens that setup.

setlocal
set "REPO=%~dp0..\.."

if exist "%REPO%\pyproject.toml" (
    where uv >nul 2>nul
    if errorlevel 1 goto :no_uv
    pushd "%REPO%"
    uv run thermosim gui %*
    set "RC=%ERRORLEVEL%"
    popd
    goto :done
)

where thermosim >nul 2>nul
if errorlevel 1 goto :not_installed
thermosim gui %*
set "RC=%ERRORLEVEL%"
goto :done

:no_uv
echo.
echo ThermoSim needs "uv" (a free Python tool), and it was not found.
echo Install it by opening PowerShell and running:
echo.
echo     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 ^| iex"
echo.
echo Then close that window and double-click this file again.
set "RC=1"
goto :done

:not_installed
echo.
echo ThermoSim is not installed. See packaging\windows\README.md for the steps.
set "RC=1"

:done
if not "%RC%"=="0" (
    echo.
    echo ThermoSim stopped with an error ^(exit code %RC%^).
    pause
)
endlocal & exit /b %RC%
