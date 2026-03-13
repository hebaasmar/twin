# Twin

Real-time meeting assistant with live transcription and AI-powered responses.

## Features

- Live audio transcription with speaker diarization
- Real-time AI responses via server-sent events
- Session management for organizing conversations
- Context-aware knowledge base retrieval

## Setup

1. Install dependencies:
   ```bash
   pip install flask anthropic deepgram-sdk numpy
   ```

2. Set environment variables:
   ```bash
   export ANTHROPIC_API_KEY=your_key
   export DEEPGRAM_API_KEY=your_key
   ```

3. Run:
   ```bash
   python overlay.py
   ```

4. Open http://localhost:5056

## Usage

- Click **Start Meeting** to begin a session
- Hold **spacebar** to capture audio
- View transcription and AI responses in real-time
- Use the text input to steer the assistant's focus
