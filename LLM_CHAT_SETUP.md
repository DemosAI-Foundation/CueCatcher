# LLM Chat Setup for CueCatcher

## Overview

CueCatcher now includes a real-time chat interface that lets you ask questions about the child's communication patterns using your local LLM (llama.cpp).

## Architecture

- **Database**: SQLite (default) - stores session data locally
- **LLM Backend**: llama.cpp running at `http://127.0.0.1:8083`
- **Chat UI**: New "Ask AI" tab in the interface
- **API Endpoint**: `POST /api/chat`

## Setup Instructions

### 1. Start llama.cpp Server

```bash
# Download llama.cpp if you haven't already
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Start the server with your model
./server -m /path/to/your-model.gguf \
         --host 127.0.0.1 \
         --port 8083 \
         --ctx-size 4096 \
         --n-gpu-layers 35  # Adjust based on your GPU
```

**Recommended models:**
- `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (fast, good for chat)
- `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` (better reasoning)
- `Phi-3-mini-4k-instruct-Q4_K_M.gguf` (very fast)

### 2. Configure CueCatcher

The system is now configured to use SQLite by default. No additional configuration needed!

Optional: Edit `/workspace/config/settings.py` if you want to change the database path:
```python
db_url: str = "sqlite:///data/cuecatcher.db"
```

### 3. Start CueCatcher

```bash
cd /workspace
python -m server.main
```

### 4. Open the UI

Navigate to `http://localhost:8084` in your browser.

You'll see a new **"🧭 Ask AI"** tab in the navigation.

## Using the Chat

### Example Questions

Once a session is active, you can ask:

- "What communication patterns do you see?"
- "How many gaze alternations occurred?"
- "Is the child showing signs of intentional communication?"
- "What strategies would help encourage more communication?"
- "Compare this session to previous ones" (longitudinal analysis)

### Features

- **Streaming responses**: See the LLM's answer as it's generated
- **Session context**: Automatically includes current session data
- **Conversation history**: Remembers previous questions in the chat
- **Connection status**: Shows whether llama.cpp is connected

## How It Works

1. **Data Collection**: Session data is stored in SQLite (`data/cuecatcher.db`) and JSON files (`data/sessions/`)

2. **Context Building**: When you ask a question, the system:
   - Loads session summary (duration, episodes, interpretations)
   - Extracts communication highlights (gaze alternations, coordinated signals)
   - Identifies behavioral patterns
   - Builds a structured prompt for the LLM

3. **LLM Analysis**: The llama.cpp backend processes your question with the session context

4. **Streaming Response**: Tokens are streamed back to the UI in real-time

## API Reference

### Chat Endpoint

```http
POST /api/chat
Content-Type: application/json

{
  "message": "What communication patterns do you see?",
  "session_id": "optional-session-id"
}
```

Response (Server-Sent Events):
```
data: {"token": "Based"}
data: {"token": " on"}
data: {"token": " the"}
data: {"token": " session"}
...
data: [DONE]
```

## Troubleshooting

### "LLM Offline" Message

1. Check if llama.cpp is running:
   ```bash
   curl http://127.0.0.1:8083/health
   ```

2. Make sure you started the server with the correct port:
   ```bash
   ./server -m model.gguf --host 127.0.0.1 --port 8083
   ```

### Database Errors

If you see SQLite errors:
1. Check that `data/` directory exists and is writable
2. Try deleting `data/cuecatcher.db` to start fresh
3. The system will work without a database (uses JSON files only)

### Slow Responses

- Use a smaller/faster model (e.g., Phi-3, Llama-3.2-3B)
- Reduce `--ctx-size` if memory is limited
- Increase `--n-gpu-layers` if you have a powerful GPU

## Privacy Note

All processing happens **locally** on your machine:
- No data leaves your computer
- Session data stays in SQLite/JSON files
- LLM runs entirely offline with llama.cpp

## Next Steps

1. Start a session by clicking "▶ Start"
2. Click the "🧭 Ask AI" tab
3. Ask questions about the child's communication!

Example: *"I noticed the child looking between me and the toy. What does this mean?"*
