@echo off
rem ============================================================
rem  IMPORTANT: keep this file PURE ASCII (see embed_2025.bat for why).
rem  cmd.exe mis-parses multi-byte characters once `chcp 65001` runs.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

set "VENV=qwen3-embedding\.venv\Scripts\python.exe"
if not exist "%VENV%" (
  echo [ERROR] venv python not found:
  echo         %VENV%
  pause
  exit /b 1
)

echo ============================================================
echo   Rebuilding vectors for files that already have one
echo.
echo   Use this after the table-extraction fix changed the chunks:
echo   stale vectors are re-computed, valid ones are skipped.
echo.
echo   * Resumable: Ctrl+C anytime, progress is kept per report
echo   * Needs about 6 GB RAM - keep other apps closed
echo ============================================================
echo.

"%VENV%" -u "rag\embed.py" --existing

echo.
echo ============================================================
echo   Finished.
echo   Next: double-click build_index.bat, then run_rag.bat
echo ============================================================
pause
