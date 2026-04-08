@echo off
title CueCatcher Server
color 0A

echo Starting CueCatcher...
echo.
cd /d "C:\Users\bbonn\MyMachine\CueCatcher"
call venv\Scripts\activate.bat

:: Set environment
set CueCatcher_REDIS_URL=redis://localhost:6379/0
set CueCatcher_DB_URL=sqlite+aiosqlite:///data/CueCatcher.db
set CueCatcher_DEVICE=cuda:0
set CueCatcher_MODEL_DIR=C:\Users\bbonn\MyMachine\CueCatcher\models

:: Start server
echo CueCatcher is starting on http://127.0.0.1:8084
echo Open this URL in your browser: http://localhost:8084
echo.
echo Press Ctrl+C to stop.
echo.

python -m uvicorn server.main:app --host 127.0.0.1 --port 8084 --ws-max-size 16777216

pause