@echo off
setlocal

cd /d "%~dp0"

if not exist ".uv-cache" mkdir ".uv-cache"
set UV_CACHE_DIR=%CD%\.uv-cache

if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv\Scripts\python.exe
    echo Run: uv sync --extra dev
    exit /b 1
)

uv run --no-sync python scripts\run_full_analysis.py %*
exit /b %ERRORLEVEL%
