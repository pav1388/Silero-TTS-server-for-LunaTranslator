@echo off
REM set DEBUG=1
set TORCH_DEVICE=cpu
cd ..
python silero-tts-rt-server.py
pause