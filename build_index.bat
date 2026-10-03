@echo off
rem ============================================================
rem  IMPORTANT: keep this file PURE ASCII (see embed_2025.bat for why).
rem  cmd.exe mis-parses multi-byte characters once `chcp 65001` runs.
rem  All Chinese output comes from the Python scripts.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

set "VENV=qwen3-embedding\.venv\Scripts\python.exe"
set "PY=python"
if not exist "%VENV%" (
  echo [ERROR] venv python not found:
  echo         %VENV%
  pause
  exit /b 1
)

echo ============================================================
echo  [1/3] Embedding        (already-done reports are skipped)
echo ============================================================
"%VENV%" "rag\embed.py" %EMB_ARGS%
if errorlevel 1 goto fail

echo.
echo ============================================================
echo  [2/3] Merging vector matrix
echo ============================================================
"%PY%" "rag\merge_vectors.py"
if errorlevel 1 goto fail

echo.
echo ============================================================
echo  [3/3] UMAP + BM25 + byte-offset table
echo ============================================================
"%PY%" "rag\build_viz_index.py"
if errorlevel 1 goto fail

echo.
echo [DONE] Index ready. Now double-click run_rag.bat
pause
exit /b 0

:fail
echo.
echo [FAILED] The step above returned an error. See the output.
pause
exit /b 1
