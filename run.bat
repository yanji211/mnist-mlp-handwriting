@echo off
cd /d "%~dp0"
if exist "%~dp0runtime\python.exe" (
  echo 正在用内置 Python 启动手写数字识别服务（便携版，无需安装任何环境）...
  "%~dp0runtime\python.exe" "%~dp0app.py" --port 8011
) else (
  echo 正在用系统 Python 启动（请先确保已 pip install numpy）...
  python app.py --port 8011
)
pause
