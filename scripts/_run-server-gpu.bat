@echo off
REM set DEBUG=1
set TORCH_DEVICE=cuda
cd ..
python silero-tts-rt-server.py
pause