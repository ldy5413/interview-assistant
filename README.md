# Real-time Speech to Text Application

This is a real-time speech-to-text application that uses OpenAI's Whisper model to transcribe audio from both your microphone and system audio output.

## Requirements

- Python 3.7 or higher
- A working microphone
- Windows/Linux/MacOS

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python stt_app.py
```

2. Select your audio source:
   - Microphone: Captures audio from your default microphone
   - System Audio: Captures audio from your system output (e.g., speakers)
   - Both: Captures from both sources simultaneously

3. Choose your language:
   - English: For English audio input
   - Chinese: For Chinese (Mandarin) audio input

4. Select Whisper Model:
   - Tiny: Fastest but least accurate
   - Base: Good balance of speed and accuracy
   - Small: Better accuracy than base, slightly slower
   - Medium: High accuracy, slower processing
   - Large: Best accuracy, but slowest processing

5. Click "Start Streaming" to begin transcription
6. The transcribed text will appear in real-time in the respective text areas
7. Click "Stop Streaming" to end the session

## Features

- Real-time audio streaming and transcription
- Support for both microphone input and system audio output
- Multiple Whisper model options for different accuracy/speed trade-offs
- Dual-window interface showing transcriptions from both sources
- Language support for English and Chinese
- Real-time language switching during transcription
- Model loading status indicator

## Notes

- The application uses Whisper models of varying sizes:
  - Tiny: ~39M parameters
  - Base: ~74M parameters
  - Small: ~244M parameters
  - Medium: ~769M parameters
  - Large: ~1.5B parameters
- The first time you run the application with a particular model, it will download that model (this may take a few minutes)
- Larger models provide better accuracy but require more processing power and memory
- Audio is processed in 3-second chunks for real-time performance
- System audio capture requires appropriate system permissions
- Language can be switched in real-time without stopping the stream
- Model selection is only available before starting the stream 