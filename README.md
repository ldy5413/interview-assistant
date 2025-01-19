# Real-time Speech to Text Application

This is a real-time speech-to-text application that uses OpenAI's Whisper model to transcribe audio from your microphone and provides real-time translation and AI-powered question answering.

## Requirements

- Python 3.7 or higher
- A working microphone
- Windows/Linux/MacOS
- OpenAI API key (for translation and question answering)

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

3. Configure your OpenAI API key (optional, for translation and AI features):
   - Create a `.env` file in the project root
   - Add your API key: `OPENAI_API_KEY=your-key-here`
   - Or configure it through the UI in the API Settings

## Usage

1. Run the application:
```bash
python stt_app.py
```

2. Configure the transcription:
   - Select your language (English/Chinese)
   - Choose a Whisper model
   - Enable/disable auto-translation
   - Enable/disable AI question answering
   - Select an AI role for answering questions

3. Select Whisper Model:
   - Tiny: Fastest but least accurate
   - Base: Good balance of speed and accuracy
   - Small: Better accuracy than base, slightly slower
   - Medium: High accuracy, slower processing
   - Large: Best accuracy, but slowest processing
   - Faster variants available for each model size

4. Choose AI Assistant Role (when AI answering is enabled):
   - General Assistant: Balanced, general-purpose responses
   - Technical Interviewer: Focus on technical accuracy and depth
   - HR Interviewer: Focus on soft skills and behavioral aspects
   - Meeting Participant: Helps clarify and summarize discussions
   - Student: Educational context from a student's perspective
   - Teacher: Educational explanations with teaching context
   - Custom: Define your own AI role behavior

5. Click "Start Streaming" to begin transcription
6. The interface shows three panels:
   - Original transcription (streaming word by word)
   - Real-time translation (if enabled)
   - AI answers to questions (if enabled)
7. Click "Stop Streaming" to end the session

## Features

- Real-time word-by-word transcription display
- Automatic sentence detection and formatting
- Support for both English and Chinese languages
- Real-time translation between English and Chinese
- AI-powered question detection and answering
- Customizable AI assistant roles
- Multiple Whisper model options for different accuracy/speed trade-offs
- Real-time language switching during transcription
- Model loading status indicator
- Export and load conversation history
- Conversation context awareness for AI answers

## Notes

- The application uses Whisper models of varying sizes:
  - Tiny: ~39M parameters
  - Base: ~74M parameters
  - Small: ~244M parameters
  - Medium: ~769M parameters
  - Large: ~1.5B parameters
- The first time you run the application with a particular model, it will download that model (this may take a few minutes)
- Larger models provide better accuracy but require more processing power and memory
- Audio is processed in real-time with word-by-word display
- AI answers consider recent conversation context for more relevant responses
- Translation and AI features require a valid OpenAI API key
- Language can be switched in real-time without stopping the stream
- Model selection is only available before starting the stream 