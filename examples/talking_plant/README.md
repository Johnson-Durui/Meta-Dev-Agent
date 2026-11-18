# Talking Plant — Voice & Emotion Interactive Arduino Companion

A hardware-coupled AI companion curated as a showcase output in `Meta-Dev-Agent`. It listens to live microphone audio via `faster-whisper`, classifies speaker emotion and conversational intent with `sentence-transformers`, replies via TTS, and drives dual servos + a 27-LED NeoPixel ring over USB serial.

## Project Layout

- `talking_plant.py` — Host Python runtime (streaming audio capture, Whisper ASR, prototype emotion/intent classifier, TTS, serial protocol).
- `talking_plant_arduino/talking_plant_arduino.ino` — Arduino firmware handling `LED`, `FLASH`, `SERVO`, `NOD`, `WIGGLE`, `PING`, and `SLEEP` serial commands.
- `requirements.txt` — Python dependencies isolated to this hardware showcase.

## Quick Start

1. Flash `talking_plant_arduino/talking_plant_arduino.ino` to an Arduino board wired with NeoPixel (`D6`) and dual servos (`D9`, `D10`).
2. Install dependencies in a virtual environment:
   ```bash
   pip install -r examples/talking_plant/requirements.txt
   ```
3. Configure serial port and launch:
   ```bash
   export PLANT_SERIAL_PORT="/dev/tty.usbmodem14101"  # or COM3 on Windows
   python examples/talking_plant/talking_plant.py
   ```
