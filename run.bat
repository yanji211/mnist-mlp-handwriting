@echo off
cd /d "%~dp0"

:: === Startup checks ===
if not exist "%~dp0runtime\python.exe" (
    echo [ERROR] runtime\python.exe not found
    pause
    exit /b 1
)

if not exist "%~dp0model.npz" (
    echo [ERROR] model.npz not found - run train.py first
    pause
    exit /b 1
)

:: CNN model is optional but load if present
set CNN_FLAG=
if exist "%~dp0cnn_model.npz" (
    echo [OK] CNN model found - MLP vs CNN comparison enabled
    set CNN_FLAG=--cnn "%~dp0cnn_model.npz"
) else (
    echo [INFO] cnn_model.npz not found - MLP only
)

:: Kill any old process holding port 8011
echo.
echo Cleaning port 8011...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8011" ^| findstr "LISTENING"') do (
    echo   Killing PID %%a
    taskkill /F /PID %%a >nul 2>nul
)

echo.
echo Starting MNIST server on http://localhost:8011 ...
"%~dp0runtime\python.exe" "%~dp0app.py" --port 8011 %CNN_FLAG%
pause