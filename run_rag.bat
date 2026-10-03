@echo off
rem ============================================================
rem  IMPORTANT: keep this file PURE ASCII (see embed_2025.bat for why).
rem  cmd.exe mis-parses multi-byte characters once `chcp 65001` runs.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

rem ---- optional: read deepseek.env (KEY=VALUE per line) if present ----
if exist "deepseek.env" (
  for /f "usebackq tokens=1,* delims==" %%a in ("deepseek.env") do (
    if not "%%a"=="" set "%%a=%%b"
  )
)

set "PY=python"
rem %LOCALAPPDATA% expands to ...\AppData\Local - keeps this file pure ASCII
where python >nul 2>nul || set "PY=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"

echo Starting RAG visualiser ...
"%PY%" "rag\app.py" %*
pause
