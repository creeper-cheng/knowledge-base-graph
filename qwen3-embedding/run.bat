@echo off
REM 用项目自带的 venv 跑 qwen3_embed.py，透传所有参数
REM 例: run.bat
REM     run.bat --text "你好" "Hello"
REM     run.bat --compare "猫" "狗" "量子力学"
"%~dp0.venv\Scripts\python.exe" "%~dp0qwen3_embed.py" %*
