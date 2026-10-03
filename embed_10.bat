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
echo   Embedding 10 companies (2025) - the 12-company target set
echo.
echo   601233 Tongkun     603225 Xin Feng Ming   000703 Hengyi
echo   002493 Rongsheng   000936 Huaxi          002064 Huafeng
echo   000782 Heshen      600810 Shenma          300699 Guangwei
echo   688295 Zhongfu
echo.
echo   Already done and skipped: 002254 Taihe, 000949 Xinxiang
echo   Resumable: Ctrl+C anytime, progress is kept per report.
echo   Needs about 7 GB RAM. Estimated 2.5 hours.
echo ============================================================
echo.

"%VENV%" -u "rag\embed.py" --year 2025 --only 601233,603225,000703,002493,000936,002064,000782,600810,300699,688295

echo.
echo ============================================================
echo   Finished. Next: double-click build_index.bat, then run_rag.bat
echo ============================================================
pause
