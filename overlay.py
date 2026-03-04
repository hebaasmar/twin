"""
Interview coach — Digital Twin interface.

Run:  python3 overlay.py
Then: browser opens automatically at http://localhost:5055

Features:
- Three-panel Digital Twin UI (Sessions / Stage / Agent)
- Start Meeting: Begins continuous transcription
- Spacebar: Hold to capture question, release to get Claude answer
- Agent chat: Talk to your twin before/during/after the interview
- Session persistence: All sessions saved to ~/second-brain-dev/sessions/
"""

import json
import os
import queue
import re
import struct
import sys
import threading
import uuid
import webbrowser
import wave
import tempfile
import traceback
from datetime import datetime
from typing import Optional

from flask import Flask, Response, jsonify, request
from dotenv import load_dotenv

import pyaudio
import whisper
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SESSIONS_DIR       = os.path.expanduser("~/second-brain-dev/sessions")
PORT               = 5055
BASE_DIR           = os.path.expanduser("~/second-brain-dev")
KB_CHUNKS_FILE      = os.path.join(BASE_DIR, "kb_chunks.json")
KB_EMBEDDINGS_FILE  = os.path.join(BASE_DIR, "kb_chunk_embeddings.npz")
TWEAKS_FILE         = os.path.join(BASE_DIR, "tweaks.md")
SYSTEM_PROMPT_FILE  = os.path.join(BASE_DIR, "system_prompt.md")

TWEAKS_HEADER = """Twin Response Tweaks
These rules are injected into the Twin's system prompt to improve responses over time.
Add feedback after sessions. One line per tweak, starting with "- "
"""

# Local KB index (loaded at startup from sync_kb.py output)
kb_chunks: list = []
kb_embeddings: Optional[np.ndarray] = None
embed_model = None

# Audio settings
CHUNK    = 1024
FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 16000

# ── App state ─────────────────────────────────────────────────────────────────
app          = Flask(__name__)
meeting_active = False
recording    = False
_clients: list[queue.Queue] = []
_clients_lock = threading.Lock()

# Spacebar Q&A recording state
audio_frames      = []
audio_stream      = None
audio_pyaudio     = None
record_thread     = None
record_stop_event = threading.Event()

# Continuous transcription state
continuous_active    = False
continuous_thread    = None
continuous_stop_event = threading.Event()


def _blank_session() -> dict:
    return {
        "id": None,
        "name": "New Session",
        "created_at": None,
        "meeting_context": "",
        "interviewer_name": "Speaker 1",
        "transcript": "",
        "exchanges": [],
        "agent_messages": [],
        "is_meeting_active": False,
    }


current_session: dict = _blank_session()

# Loaded at startup
whisper_model = None

# Conversation history for Claude API (cleared each meeting)
conversation_history: list = []

# Pending streaming requests: stream_id → {system_prompt, messages, ...}
_pending_streams: dict = {}
_pending_streams_lock = threading.Lock()


# ── Local KB search ───────────────────────────────────────────────────────────
def search_local_kb(query: str, top_k: int = 6) -> list:
    """Semantic search over pre-built local chunk index. No network calls."""
    if kb_embeddings is None or not kb_chunks:
        log_error("KB index not loaded. Run sync_kb.py first.")
        return []

    query_embedding = embed_model.encode([query], normalize_embeddings=True)[0]
    similarities = np.dot(kb_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = kb_chunks[idx]
        score = float(similarities[idx])
        log_info(f"Match: [{chunk['layer']}] {chunk['page_title']} (score: {score:.3f})")
        results.append(chunk)

    return results


# ── Ensure sessions directory exists ──────────────────────────────────────────
os.makedirs(SESSIONS_DIR, exist_ok=True)


# ── Logging helpers ───────────────────────────────────────────────────────────
def log_info(msg):
    print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} {msg}")

def log_error(msg):
    print(f"[ERROR] {datetime.now().strftime('%H:%M:%S')} {msg}")


# ── SSE broadcast ─────────────────────────────────────────────────────────────
def broadcast(payload: dict):
    with _clients_lock:
        for q in list(_clients):
            q.put(payload)


# ── Tweaks (feedback for Twin system prompt) ───────────────────────────────────
def _read_tweaks_file() -> str:
    """Return full contents of tweaks.md or empty string."""
    try:
        with open(TWEAKS_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _count_tweaks(content: str) -> int:
    """Count lines that are tweak bullets (start with '- ')."""
    return sum(1 for line in content.splitlines() if line.strip().startswith("- "))


def load_tweaks_for_prompt() -> str:
    """Return tweaks file content for injection into system prompt."""
    return _read_tweaks_file()


def append_tweak(text: str) -> int:
    """Append one tweak line to tweaks.md. Returns total tweak count after append."""
    line = (text or "").strip()
    if not line:
        return _count_tweaks(_read_tweaks_file())
    with open(TWEAKS_FILE, "a", encoding="utf-8") as f:
        f.write("\n- " + line + "\n")
    return _count_tweaks(_read_tweaks_file())


def clear_tweaks() -> None:
    """Rewrite tweaks.md to header only."""
    with open(TWEAKS_FILE, "w", encoding="utf-8") as f:
        f.write(TWEAKS_HEADER.strip() + "\n")


# ── Session management ────────────────────────────────────────────────────────
def extract_interviewer_name(text: str) -> str:
    """Try to extract an interviewer's first name from meeting context text."""
    patterns = [
        r"talking to (\w+)",
        r"meeting with (\w+)",
        r"interviewing with (\w+)",
        r"interview with (\w+)",
        r"interviewer is (\w+)",
        r"speaking with (\w+)",
        r"chat with (\w+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            name = m.group(1)
            if name.lower() not in ("the", "a", "an", "my", "our"):
                return name.capitalize()
    return "Speaker 1"


def start_new_session(session_id: Optional[str] = None) -> dict:
    global current_session, conversation_history
    now = datetime.now()
    sid = session_id or str(uuid.uuid4())[:8]
    current_session = {
        "id": sid,
        "name": f"Session {now.strftime('%b %d, %Y')}",
        "created_at": now.isoformat(),
        "meeting_context": "",
        "interviewer_name": "Speaker 1",
        "transcript": "",
        "exchanges": [],
        "agent_messages": [],
        "is_meeting_active": True,
    }
    conversation_history = []
    save_session()
    log_info(f"Session started: {current_session['id']}")
    return current_session


def save_session():
    if not current_session.get("id"):
        return
    filepath = os.path.join(SESSIONS_DIR, f"{current_session['id']}.json")
    with open(filepath, "w") as f:
        json.dump(current_session, f, indent=2)


def save_session_data(session_data: dict):
    """Save an arbitrary session dict to disk."""
    sid = session_data.get("id")
    if not sid:
        return
    filepath = os.path.join(SESSIONS_DIR, f"{sid}.json")
    with open(filepath, "w") as f:
        json.dump(session_data, f, indent=2)


def list_sessions() -> list:
    sessions = []
    if os.path.exists(SESSIONS_DIR):
        for fname in os.listdir(SESSIONS_DIR):
            if fname.endswith(".json"):
                fpath = os.path.join(SESSIONS_DIR, fname)
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    sessions.append({
                        "id": data.get("id", fname.replace(".json", "")),
                        "name": data.get("name", "Session"),
                        "created_at": data.get("created_at", data.get("started_at", "")),
                        "exchange_count": len(data.get("exchanges", data.get("qa_pairs", []))),
                        "is_meeting_active": data.get("is_meeting_active", False),
                    })
                except Exception:
                    pass
    sessions.sort(key=lambda x: x["created_at"], reverse=True)
    return sessions


def load_session(session_id: str) -> Optional[dict]:
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None


# ── Continuous Transcription ──────────────────────────────────────────────────
def start_continuous_transcription():
    """Start continuous mic transcription in background."""
    global continuous_active, continuous_thread, continuous_stop_event

    continuous_stop_event.clear()
    continuous_active = True

    def transcribe_loop():
        SEGMENT_SECONDS = 3
        frames_per_segment = int(RATE / CHUNK * SEGMENT_SECONDS)

        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK,
            )
            log_info("Continuous transcription started...")

            frames = []
            while not continuous_stop_event.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)

                    if len(frames) >= frames_per_segment:
                        threading.Thread(
                            target=transcribe_segment,
                            args=(list(frames),),
                            daemon=True
                        ).start()
                        frames = []

                except Exception as e:
                    log_error(f"Continuous audio read error: {e}")
                    break

            stream.stop_stream()
            stream.close()
            p.terminate()
            log_info("Continuous transcription stopped.")

        except Exception as e:
            log_error(f"Continuous transcription error: {e}")
            traceback.print_exc()

    continuous_thread = threading.Thread(target=transcribe_loop, daemon=True)
    continuous_thread.start()


def transcribe_segment(frames):
    """Transcribe a segment of audio and broadcast to transcript panel."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        p = pyaudio.PyAudio()
        wf = wave.open(tmp_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        p.terminate()

        # Silence detection: skip Whisper on silent chunks to avoid hallucinations
        with wave.open(tmp_path, 'rb') as wf_check:
            raw = wf_check.readframes(wf_check.getnframes())
        if raw:
            samples = struct.unpack(f'<{len(raw) // 2}h', raw)
            rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
            if rms < 500:
                os.unlink(tmp_path)
                log_info(f"Skipping silent segment (RMS={rms:.0f})")
                return

        try:
            result = whisper_model.transcribe(tmp_path)
            text = result["text"].strip()
        except RuntimeError as e:
            log_info(f"Whisper skipped short segment: {e}")
            os.unlink(tmp_path)
            return
        os.unlink(tmp_path)

        if text and len(text) > 2:
            # Normalize: transcript may be a legacy list (old session format) or a string
            existing = current_session.get("transcript", "")
            if isinstance(existing, list):
                existing = " ".join(
                    t.get("text", "") if isinstance(t, dict) else str(t)
                    for t in existing
                )
            current_session["transcript"] = existing + text + " "
            broadcast({"type": "transcript", "text": text})

    except Exception as e:
        log_error(f"Segment transcription error: {e}")
        traceback.print_exc()


def stop_continuous_transcription():
    """Stop continuous transcription."""
    global continuous_active, continuous_stop_event
    continuous_stop_event.set()
    continuous_active = False


# ── Spacebar Q&A Recording ────────────────────────────────────────────────────
def start_recording():
    global audio_frames, audio_stream, audio_pyaudio, record_thread, record_stop_event

    audio_frames = []
    record_stop_event.clear()

    def record_loop():
        global audio_pyaudio, audio_stream, audio_frames
        try:
            audio_pyaudio = pyaudio.PyAudio()
            audio_stream = audio_pyaudio.open(
                format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK,
            )
            log_info("Q&A Recording started...")

            while not record_stop_event.is_set():
                try:
                    data = audio_stream.read(CHUNK, exception_on_overflow=False)
                    audio_frames.append(data)
                except Exception as e:
                    log_error(f"Audio read error: {e}")
                    break

            audio_stream.stop_stream()
            audio_stream.close()
            audio_pyaudio.terminate()
            log_info(f"Q&A Recording stopped. {len(audio_frames)} frames captured.")

        except Exception as e:
            log_error(f"Recording error: {e}")
            traceback.print_exc()

    record_thread = threading.Thread(target=record_loop, daemon=True)
    record_thread.start()


def stop_recording_and_transcribe():
    global record_stop_event, record_thread, audio_frames

    record_stop_event.set()
    if record_thread:
        record_thread.join(timeout=2)

    if not audio_frames:
        log_error("No audio frames captured")
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        p = pyaudio.PyAudio()
        wf = wave.open(tmp_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()
        p.terminate()

        log_info(f"Transcribing {len(audio_frames)} frames...")
        result = whisper_model.transcribe(tmp_path)
        text = result["text"].strip()

        os.unlink(tmp_path)
        log_info(f"Transcribed: {text[:80]}...")
        return text

    except Exception as e:
        log_error(f"Transcription error: {e}")
        traceback.print_exc()
        return None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return HTML


@app.route("/stream")
def stream():
    q: queue.Queue = queue.Queue()
    with _clients_lock:
        _clients.append(q)

    def generate():
        try:
            while True:
                try:
                    data = q.get(timeout=20)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _clients_lock:
                if q in _clients:
                    _clients.remove(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/current_session")
def api_current_session():
    return jsonify({
        "session_id": current_session.get("id"),
        "meeting_active": meeting_active,
    })


@app.route("/api/sessions", methods=["GET"])
def api_sessions_list():
    return jsonify(list_sessions())


@app.route("/api/sessions", methods=["POST"])
def api_sessions_create():
    global current_session
    now = datetime.now()
    sid = str(uuid.uuid4())[:8]
    session = {
        "id": sid,
        "name": f"Session {now.strftime('%b %d, %Y')}",
        "created_at": now.isoformat(),
        "meeting_context": "",
        "interviewer_name": "Speaker 1",
        "transcript": "",
        "exchanges": [],
        "agent_messages": [],
        "is_meeting_active": False,
    }
    save_session_data(session)
    if not meeting_active:
        current_session = session
    broadcast({"type": "session_update"})
    return jsonify(session)


@app.route("/api/sessions/<session_id>", methods=["GET"])
def api_session_get(session_id):
    data = load_session(session_id)
    if data:
        return jsonify(data)
    return jsonify({"error": "Session not found"}), 404


@app.route("/api/sessions/<session_id>", methods=["PUT"])
def api_session_update(session_id):
    updates = request.get_json() or {}
    data = load_session(session_id)
    if not data:
        return jsonify({"error": "Session not found"}), 404
    data.update(updates)
    save_session_data(data)
    if session_id == current_session.get("id"):
        for k, v in updates.items():
            current_session[k] = v
    broadcast({"type": "session_update"})
    return jsonify(data)


@app.route("/rename_session", methods=["POST"])
def rename_session_route():
    data       = request.get_json() or {}
    session_id = data.get("session_id", "").strip()
    new_name   = data.get("new_name", "").strip()

    if not session_id or not new_name:
        return jsonify({"ok": False, "error": "Missing session_id or new_name"}), 400

    # Sanitize: strip filesystem-unsafe characters
    safe_name = re.sub(r'[\/\\:*?"<>|]', '', new_name).strip()[:80]
    if not safe_name:
        return jsonify({"ok": False, "error": "Invalid name"}), 400

    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(filepath):
        return jsonify({"ok": False, "error": "Session not found"}), 404

    try:
        with open(filepath) as f:
            session_data = json.load(f)
        session_data["name"] = safe_name
        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)

        # Update in-memory current session if it matches
        if current_session.get("id") == session_id:
            current_session["name"] = safe_name

        log_info(f"Session {session_id} renamed to: {safe_name}")
        return jsonify({"ok": True, "name": safe_name})
    except Exception as ex:
        log_error(f"Rename error: {ex}")
        return jsonify({"ok": False, "error": str(ex)}), 500


@app.route("/delete_session", methods=["POST"])
def delete_session_route():
    data       = request.get_json() or {}
    session_id = data.get("session_id", "").strip()

    if not session_id:
        return jsonify({"ok": False, "error": "Missing session_id"}), 400

    # Guard: don't allow path traversal
    if "/" in session_id or "\\" in session_id or ".." in session_id:
        return jsonify({"ok": False, "error": "Invalid session_id"}), 400

    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(filepath):
        return jsonify({"ok": False, "error": "Session not found"}), 404

    try:
        os.remove(filepath)
        log_info(f"Session deleted: {session_id}")
        return jsonify({"ok": True})
    except Exception as ex:
        log_error(f"Delete error: {ex}")
        return jsonify({"ok": False, "error": str(ex)}), 500


@app.route("/start_meeting", methods=["POST"])
def start_meeting_route():
    global meeting_active, current_session, conversation_history

    if meeting_active:
        return jsonify({"active": True, "error": "Meeting already active"})

    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")

    if session_id:
        loaded = load_session(session_id)
        if loaded:
            # Normalize legacy transcript (old sessions stored it as a list)
            if isinstance(loaded.get("transcript"), list):
                loaded["transcript"] = " ".join(
                    t.get("text", "") if isinstance(t, dict) else str(t)
                    for t in loaded["transcript"]
                )
            current_session = loaded

    if not current_session.get("id"):
        start_new_session()

    # Reset transcript for a clean meeting start
    current_session["transcript"] = ""
    meeting_active = True
    current_session["is_meeting_active"] = True
    conversation_history = []
    start_continuous_transcription()
    save_session()
    broadcast({"type": "meeting_started", "session_id": current_session["id"]})
    log_info("Meeting started - continuous transcription activated")

    return jsonify({"active": True, "session_id": current_session.get("id")})


@app.route("/stop_meeting", methods=["POST"])
def stop_meeting_route():
    global meeting_active, conversation_history

    if not meeting_active:
        return jsonify({"active": False})

    meeting_active = False
    stop_continuous_transcription()
    conversation_history = []
    current_session["is_meeting_active"] = False
    save_session()
    broadcast({"type": "meeting_stopped", "session_id": current_session.get("id")})
    broadcast({"type": "session_update"})
    log_info("Meeting stopped - session saved, conversation history cleared")

    return jsonify({"active": False, "session_id": current_session.get("id")})


@app.route("/start_recording", methods=["POST"])
def start_recording_route():
    global recording

    if not meeting_active:
        return jsonify({"error": "Meeting not active"})

    if recording:
        return jsonify({"error": "Already recording"})

    recording = True
    start_recording()
    broadcast({"type": "recording_started"})

    return jsonify({"recording": True})


@app.route("/stop_recording", methods=["POST"])
def stop_recording_route():
    global recording

    if not recording:
        return jsonify({"error": "Not recording"})

    recording = False
    broadcast({"type": "recording_stopped"})

    threading.Thread(target=process_recording, daemon=True).start()

    return jsonify({"recording": False})


@app.route("/api/stream_response")
def stream_response_route():
    stream_id = request.args.get("id", "")

    with _pending_streams_lock:
        pending = _pending_streams.pop(stream_id, None)

    if not pending:
        def _not_found():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Stream not found or already consumed'})}\n\n"
        return Response(_not_found(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def generate():
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

            full_text = ""
            with client.messages.stream(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                system=pending["system_prompt"],
                messages=pending["messages"],
            ) as stream:
                for text_delta in stream.text_stream:
                    full_text += text_delta
                    yield f"data: {json.dumps({'type': 'chunk', 'text': text_delta})}\n\n"

            # Parse beats, persist session, update conversation history
            beats = [b.strip() for b in full_text.split("• ") if b.strip()]
            exchange_data = {
                "speaker": pending["speaker"],
                "speaker_name": pending["speaker_name"],
                "text": pending["question"],
                "beats": beats,
                "timestamp": pending["timestamp"],
            }
            current_session["exchanges"].append(exchange_data)
            save_session()

            conversation_history.append(pending["new_user_message"])
            conversation_history.append({"role": "assistant", "content": full_text.strip()})

            log_info(f"[STREAM] Complete — {len(beats)} beats, {len(full_text)} chars")
            yield f"data: {json.dumps({'type': 'done', 'beats': beats})}\n\n"

        except Exception as e:
            log_error(f"[STREAM] Error: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/feedback", methods=["GET", "POST", "DELETE"])
def api_feedback():
    """GET: return tweaks content and count. POST: append tweak. DELETE: clear all."""
    if request.method == "GET":
        content = _read_tweaks_file()
        return jsonify({"tweaks": content, "count": _count_tweaks(content)})
    if request.method == "POST":
        data = request.get_json() or {}
        tweak = (data.get("tweak") or "").strip()
        if not tweak:
            return jsonify({"error": "Missing or empty tweak"}), 400
        total = append_tweak(tweak)
        return jsonify({"status": "ok", "total_tweaks": total})
    if request.method == "DELETE":
        clear_tweaks()
        return jsonify({"status": "ok", "total_tweaks": 0})
    return jsonify({"error": "Method not allowed"}), 405


@app.route("/agent_chat", methods=["POST"])
def agent_chat_route():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "")

    if not message:
        return jsonify({"error": "No message"}), 400

    # Tweaks/feedback commands — handle locally, no Claude call
    if message.startswith("/tweak "):
        tweak = message[7:].strip()
        if tweak:
            total = append_tweak(tweak)
            return jsonify({"response": f"Tweak saved: {tweak}", "action": None})
        return jsonify({"error": "No tweak text"}), 400
    if message.startswith("/feedback "):
        tweak = message[9:].strip()
        if tweak:
            total = append_tweak(tweak)
            return jsonify({"response": f"Tweak saved: {tweak}", "action": None})
        return jsonify({"error": "No tweak text"}), 400
    if message.strip().lower() == "/clear-tweaks":
        clear_tweaks()
        return jsonify({"response": "All tweaks cleared.", "action": None})

    # Detect natural language commands
    msg_lower = message.lower()
    action = None
    start_kws = ["start the meeting", "start meeting", "begin the meeting", "begin meeting", "let's go", "let's start"]
    stop_kws  = ["stop the meeting", "end the meeting", "stop meeting", "end meeting"]

    if msg_lower in ("start", "begin", "go") or any(kw in msg_lower for kw in start_kws):
        action = "start_meeting"
    elif msg_lower in ("stop", "end") or any(kw in msg_lower for kw in stop_kws):
        action = "stop_meeting"
    elif "new session" in msg_lower:
        action = "new_session"

    # Load session for context
    session_data = load_session(session_id) if session_id else None
    if session_data is None:
        session_data = dict(current_session)

    # Build transcript text
    transcript_text = session_data.get("transcript", "")
    if isinstance(transcript_text, list):
        transcript_text = " ".join(
            t.get("text", "") if isinstance(t, dict) else str(t)
            for t in transcript_text
        )

    # Build exchanges summary
    exchanges_lines = []
    for ex in session_data.get("exchanges", []):
        exchanges_lines.append(f"{ex.get('speaker_name', 'Speaker')}: {ex.get('text', '')}")
        for beat in ex.get("beats", []):
            exchanges_lines.append(f"  Twin: {beat}")
    exchanges_summary = "\n".join(exchanges_lines)

    meeting_ctx  = session_data.get("meeting_context", "")
    session_name = session_data.get("name", "this session")

    system = (
        f"You are the voice of a digital twin for Heba Asmar, an experienced Principal PM "
        f"with deep AI/ML background.\n\n"
        f"Session: {session_name}\n"
        + (f"Pre-meeting notes:\n{meeting_ctx}\n\n" if meeting_ctx else "No pre-meeting context yet.\n\n")
        + (f"Transcript so far:\n{transcript_text[:1500]}\n\n" if transcript_text.strip() else "No transcript yet.\n\n")
        + (f"Exchanges so far:\n{exchanges_summary[:1500]}\n\n" if exchanges_summary else "No exchanges yet.\n\n")
        + "Be concise. Short sentences. Write like Slack, not like a document. Never use markdown formatting: "
        + "no ** for bold, no * for bullets, no # for headers, no numbered lists. Plain text only. "
        + "When the user pastes context or prep notes, confirm in 1-2 sentences. Don't repeat it back. "
        + "Don't summarize it. When the user asks a question, answer it directly. "
        + "Go deep only when the question requires depth.\n\n"
        + "Before a meeting: help Heba prepare. Strategy, key points to hit, relevant numbers.\n"
        + "During a meeting: you can see the transcript. Stay quiet unless she types something. "
        + "If she asks, be concise. She's mid-conversation.\n"
        + "After a meeting: debrief honestly. What went well. What to sharpen. Reference actual exchanges.\n\n"
        + "Be direct. No filler. Reference real content from the session when it's available. Don't be a cheerleader."
    )

    agent_messages = list(session_data.get("agent_messages", []))
    messages = agent_messages + [{"role": "user", "content": message}]

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=400,
            system=system,
            messages=messages,
        )
        reply = resp.content[0].text.strip()
    except Exception as e:
        log_error(f"Agent chat error: {e}")
        reply = "Sorry, couldn't reach the API right now."

    # Update agent_messages
    session_data["agent_messages"] = agent_messages + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": reply},
    ]

    # Accumulate meeting_context from pre-meeting chat
    if not meeting_active and session_id == current_session.get("id"):
        ctx = session_data.get("meeting_context", "")
        new_ctx = (ctx + f"\nUser: {message}\nTwin: {reply}").strip()
        session_data["meeting_context"] = new_ctx
        current_session["meeting_context"] = new_ctx
        # Try to extract interviewer name
        name = extract_interviewer_name(new_ctx)
        if name != "Speaker 1":
            session_data["interviewer_name"] = name
            current_session["interviewer_name"] = name

    # Persist
    save_session_data(session_data)
    if session_id == current_session.get("id"):
        current_session["agent_messages"] = session_data["agent_messages"]

    return jsonify({"response": reply, "action": action})


def filter_transcript(raw: str, max_chars: int = 2000) -> str:
    """Clean Whisper hallucinations and trim to recent content."""
    import re as _re

    # Pattern: dotted-number hallucinations like "1.0.1.1.1" or "0.0.0.0"
    dotted_numbers = _re.compile(r'^[\d\.]+$')

    # Split into segments (Whisper appends with spaces, not newlines)
    # Treat runs of 80+ chars without punctuation as a single segment
    segments = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', raw) if s.strip()]

    cleaned = []
    for seg in segments:
        # Drop pure dotted-number lines
        if dotted_numbers.match(seg.replace(' ', '')):
            continue

        # Drop segments where a short phrase (≤6 words) repeats more than 3×
        words = seg.split()
        is_repetitive = False
        for phrase_len in range(2, 7):
            if len(words) < phrase_len * 4:
                continue
            phrase = ' '.join(words[:phrase_len]).lower()
            count = seg.lower().count(phrase)
            if count > 3:
                is_repetitive = True
                break
        if is_repetitive:
            continue

        cleaned.append(seg)

    result = ' '.join(cleaned)

    # Trim to last max_chars characters (keep the most recent content)
    if len(result) > max_chars:
        result = result[-max_chars:]
        # Don't start mid-word
        first_space = result.find(' ')
        if first_space != -1:
            result = result[first_space + 1:]

    return result.strip()


def process_recording():
    """Transcribe recording, run RAG, call Claude, broadcast exchange."""
    try:
        log_info("=== PROCESSING RECORDING ===")

        # Step 1: Transcribe
        log_info("[STEP 1] Transcribing audio...")
        question = stop_recording_and_transcribe()

        if not question:
            log_error("[STEP 1] No transcription result")
            broadcast({"type": "error", "message": "No audio captured"})
            broadcast({"type": "status", "state": "listening"})
            return

        log_info(f"[STEP 1] Question: {question}")

        # Normalize speech-to-text misspellings
        corrections = {
            "cunic": "kunik", "coonik": "kunik", "kunick": "kunik",
            "imber": "imbr", "ember": "imbr", "inver": "imbr",
            "sahjal": "sajal", "sujal": "sajal",
        }
        question_normalized = question.lower()
        for wrong, right in corrections.items():
            question_normalized = question_normalized.replace(wrong, right)
        question = question_normalized

        # Step 2: Search local KB
        log_info("[STEP 2] Searching local KB...")
        results = search_local_kb(question, top_k=6)

        if not results:
            log_error("[STEP 2] No KB results")
            broadcast({
                "type": "exchange",
                "speaker": "interviewer",
                "speaker_name": current_session.get("interviewer_name", "Speaker 1"),
                "text": question,
                "beats": ["No matching content found in local KB. Run sync_kb.py to rebuild the index."],
                "error": True
            })
            broadcast({"type": "status", "state": "listening"})
            return

        for i, r in enumerate(results):
            log_info(f"[STEP 2] Result {i+1}: [{r.get('layer','?')}] {r.get('page_title','?')} ({len(r.get('text',''))} chars)")

        # Build context with layer labels so Claude knows story vs reasoning protocol
        context = "\n---\n".join([
            f"[{r['layer']} | {r['type']}] {r['page_title']}\n{r['text']}"
            for r in results
        ])
        log_info(f"[STEP 2] Context: {len(context)} chars")

        # Step 3: Build Claude messages and queue for streaming
        log_info("[STEP 3] Preparing stream...")

        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

        # Inject response tweaks (learned from past sessions) — after main prompt, before KB
        tweaks_content = load_tweaks_for_prompt()
        if tweaks_content.strip():
            system_prompt += "\n\nResponse Tweaks (learned from past sessions)\n" + tweaks_content.strip() + "\n\n"

        # Inject meeting context and recent transcript
        meeting_ctx = current_session.get("meeting_context", "")
        recent_transcript = filter_transcript(current_session.get("transcript", ""))

        if meeting_ctx and recent_transcript:
            user_content = (
                f'Meeting context:\n{meeting_ctx}\n\n'
                f'Recent conversation transcript:\n{recent_transcript}\n\n'
                f'Question: "{question}"\n\nRelevant notes:\n{context}'
            )
        elif meeting_ctx:
            user_content = (
                f'Meeting context:\n{meeting_ctx}\n\n'
                f'Question: "{question}"\n\nRelevant notes:\n{context}'
            )
        elif recent_transcript:
            user_content = (
                f'Recent conversation transcript:\n{recent_transcript}\n\n'
                f'Question: "{question}"\n\nRelevant notes:\n{context}'
            )
        else:
            user_content = f'Question: "{question}"\n\nRelevant notes:\n{context}'

        new_user_message = {"role": "user", "content": user_content}
        messages = conversation_history + [new_user_message]
        interviewer_name = current_session.get("interviewer_name", "Speaker 1")

        stream_id = str(uuid.uuid4())[:8]
        with _pending_streams_lock:
            _pending_streams[stream_id] = {
                "system_prompt": system_prompt,
                "messages": messages,
                "new_user_message": new_user_message,
                "question": question,
                "speaker": "interviewer",
                "speaker_name": interviewer_name,
                "timestamp": datetime.now().isoformat(),
            }

        log_info(f"[STEP 3] Stream queued: {stream_id} ({len(messages)} messages)")
        broadcast({
            "type": "exchange_start",
            "stream_id": stream_id,
            "speaker": "interviewer",
            "speaker_name": interviewer_name,
            "text": question,
        })
        broadcast({"type": "status", "state": "listening"})
        log_info("=== QUEUED FOR STREAMING ===")

    except Exception as e:
        log_error(f"Processing error: {e}")
        traceback.print_exc()
        broadcast({"type": "error", "message": str(e)})
        broadcast({"type": "status", "state": "listening"})


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Twin</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --void: #0C0B0A;
    --deep: #141312;
    --surface: #1A1918;
    --raised: #222120;
    --line: #2C2A28;
    --line-subtle: #232220;
    --text-bright: #E8E0D4;
    --text: #B8AFA4;
    --text-dim: #706860;
    --text-ghost: #453F3A;
    --pulse: #C4956A;
    --pulse-dim: rgba(212, 145, 90, 0.08);
    --pulse-medium: rgba(212, 145, 90, 0.15);
    --pulse-glow: rgba(212, 145, 90, 0.25);
    --teal: #4DB8A4;
    --teal-dim: rgba(77, 184, 164, 0.08);
    --teal-medium: rgba(77, 184, 164, 0.15);
    --green: #7AAA6E;
    --green-dim: rgba(122, 170, 110, 0.1);
    --red-soft: #C47A6A;
  }

  body {
    font-family: 'DM Sans', 'Outfit', sans-serif;
    background: var(--void);
    color: var(--text);
    height: 100vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  /* ═══ PRESENCE BAR ═══════════════════════════════════════════ */
  .presence {
    height: 52px;
    background: var(--void);
    border-bottom: 1px solid var(--line-subtle);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    flex-shrink: 0;
  }

  .identity {
    display: flex;
    align-items: center;
    gap: 14px;
  }

  .twin-mark {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: radial-gradient(circle at 40% 40%, var(--teal), #2A6B5E);
    box-shadow: 0 0 12px var(--teal-dim);
    flex-shrink: 0;
  }

  .twin-mark.breathing {
    animation: breathe 4s ease-in-out infinite;
  }

  @keyframes breathe {
    0%, 100% {
      box-shadow: 0 0 12px var(--teal-dim), 0 0 3px var(--teal-dim);
      transform: scale(1);
    }
    50% {
      box-shadow: 0 0 24px var(--teal-medium), 0 0 6px var(--teal-dim);
      transform: scale(1.03);
    }
  }

  .twin-name {
    font-size: 17px;
    font-weight: 500;
    color: var(--text-bright);
    letter-spacing: -0.01em;
  }

  .presence-center {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .session-viewing {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    color: var(--text-dim);
  }

  .session-viewing-name { color: var(--text); font-weight: 500; }
  .session-viewing-date { font-size: 11px; color: var(--text-ghost); }

  .badge-past {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-ghost);
    background: var(--raised);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 2px 8px;
  }

  .btn-session {
    background: none;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 6px 18px;
    font-size: 12px;
    font-weight: 500;
    font-family: 'DM Sans', 'Outfit', sans-serif;
    color: var(--text-dim);
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.02em;
  }

  .btn-session:hover { border-color: var(--text-dim); color: var(--text); }

  .btn-session.live {
    border-color: #E8956A;
    color: #E8956A;
    background: transparent;
    border-radius: 6px;
    font-size: 13px;
  }

  .btn-session.live:hover { background: rgba(232, 149, 106, 0.1); }

  .btn-session.start {
    border-color: var(--green);
    color: var(--green);
    background: rgba(122, 170, 110, 0.06);
  }

  .btn-session.start:hover { background: rgba(122, 170, 110, 0.1); }

  .capture-state {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    font-weight: 400;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  .capture-state.listening  { color: var(--green); }
  .capture-state.capturing  { color: var(--pulse); animation: capture-flash 0.6s ease-in-out infinite; }
  .capture-state.processing { color: var(--text-dim); }
  .capture-state.inactive   { color: var(--text-ghost); }

  @keyframes capture-flash { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

  .capture-dot { width: 6px; height: 6px; border-radius: 50%; }
  .listening  .capture-dot { background: var(--green); box-shadow: 0 0 8px rgba(122, 170, 110, 0.3); }
  .capturing  .capture-dot { background: var(--pulse); box-shadow: 0 0 10px var(--pulse-glow); animation: dot-beat 0.6s ease-in-out infinite; }
  .processing .capture-dot { background: var(--text-dim); animation: dot-fade 1s ease-in-out infinite; }
  .inactive   .capture-dot { background: var(--text-ghost); }

  @keyframes dot-beat { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.4); } }
  @keyframes dot-fade { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }

  /* ── Toggle ──────────────────────────────────────── */
  .toggle-memory {
    background: none;
    border: none;
    color: var(--text-ghost);
    cursor: pointer;
    padding: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: color 0.15s;
    margin-right: 4px;
  }

  .toggle-memory:hover { color: var(--text); }
  .toggle-memory svg { width: 16px; height: 16px; transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1); }
  .toggle-memory.is-collapsed svg { transform: rotate(180deg); }

  /* ═══ THREE SPACES ═══════════════════════════════════════════ */
  .spaces {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* ── Memory (Sessions) ───────────────────────────── */
  .memory {
    width: 220px;
    background: var(--deep);
    border-right: 1px solid var(--line-subtle);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    transition: width 0.25s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.2s ease;
    overflow: hidden;
  }

  .memory.collapsed { width: 0px; border-right: none; }
  .memory.collapsed .memory-inner { opacity: 0; pointer-events: none; }

  .memory-inner {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-width: 220px;
    opacity: 1;
    transition: opacity 0.15s ease;
  }

  .memory-header {
    padding: 18px 20px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .memory-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-ghost);
  }

  .btn-new {
    background: none;
    border: none;
    color: var(--text-dim);
    font-size: 18px;
    cursor: pointer;
    line-height: 1;
    padding: 0;
    transition: color 0.15s;
    font-family: 'DM Sans', 'Outfit', sans-serif;
  }

  .btn-new:hover { color: var(--pulse); }

  .memory-list { flex: 1; overflow-y: auto; padding: 0 10px; }

  .mem-item {
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s;
    margin-bottom: 2px;
    border-left: 2px solid transparent;
  }

  .mem-item:hover { background: var(--raised); }
  .mem-item.active  { background: rgba(77, 184, 164, 0.12); border-left-color: #4DB8A4; }
  .mem-item.viewing { background: var(--raised); border-left-color: var(--text-dim); }

  .mem-name { font-size: 13px; font-weight: 500; color: var(--text); margin-bottom: 2px; }
  .mem-item.active .mem-name, .mem-item.viewing .mem-name { color: var(--text-bright); }
  .mem-when { font-size: 10px; color: var(--text-ghost); }

  .mem-row {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .mem-row .mem-name { flex: 1; min-width: 0; }

  .mem-actions {
    display: flex;
    gap: 2px;
    opacity: 0;
    flex-shrink: 0;
    transition: opacity 0.15s;
  }

  .mem-item:hover .mem-actions { opacity: 1; }

  .mem-icon {
    background: none;
    border: none;
    color: var(--text-ghost);
    cursor: pointer;
    padding: 2px 3px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: color 0.15s, background 0.15s;
    line-height: 1;
  }

  .mem-icon:hover { color: var(--text); background: var(--line); }
  .mem-icon.delete:hover { color: var(--red-soft); background: rgba(196, 122, 106, 0.1); }
  .mem-icon svg { width: 11px; height: 11px; }

  .mem-name-input {
    flex: 1;
    background: var(--raised);
    border: 1px solid var(--pulse);
    border-radius: 4px;
    color: var(--text-bright);
    font-size: 13px;
    font-weight: 500;
    font-family: 'DM Sans', 'Outfit', sans-serif;
    padding: 1px 6px;
    outline: none;
    min-width: 0;
  }

  /* ── The Stage ───────────────────────────────────── */
  .stage {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--surface);
    min-width: 0;
  }

  .stage-half {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
  }

  .stage-label {
    padding: 14px 28px 8px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #666;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .scroll-area { flex: 1; overflow-y: auto; padding: 0 28px 16px; }

  .transcript-line {
    font-size: 14px;
    line-height: 1.8;
    color: var(--text-dim);
    font-weight: 300;
  }

  .placeholder-text {
    font-size: 13px;
    color: var(--text-ghost);
    font-weight: 300;
    font-style: italic;
    padding-top: 4px;
  }

  .divider {
    height: 6px;
    background: transparent;
    border-top: 1px solid var(--line-subtle);
    flex-shrink: 0;
    position: relative;
    cursor: row-resize;
  }

  .divider::after {
    content: '';
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 32px;
    height: 3px;
    border-radius: 2px;
    background: #444;
    transition: background 0.15s;
  }

  .divider:hover::after { background: #4DB8A4; }

  /* ── Exchange blocks ──────────────────────────────── */
  .exchange { margin-bottom: 28px; position: relative; }

  .who {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
  }

  .who-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .who-dot.interviewer { background: var(--pulse); }
  .who-dot.heba        { background: var(--pulse); }
  .who-dot.twin        { background: var(--teal); }

  .who-name {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  .who-name.interviewer { color: var(--pulse); }
  .who-name.heba        { color: var(--pulse); }
  .who-name.twin        { color: var(--teal); }

  .who-text {
    font-size: 14px;
    line-height: 1.6;
    color: var(--text);
    font-weight: 400;
    padding-left: 18px;
    margin-bottom: 12px;
  }

  .beats { padding-left: 18px; }

  .beat {
    font-size: 13px;
    line-height: 1.65;
    font-weight: 300;
    color: var(--text-bright);
    padding: 7px 0 7px 16px;
    border-left: 1px solid var(--line);
    margin-bottom: 2px;
    transition: all 0.2s;
  }

  .beat:hover {
    border-left-color: var(--teal);
    border-left-width: 2px;
    padding-left: 15px;
    color: #F0E8DC;
  }

  /* ── Streaming beats ──────────────────────────────── */
  .beat-streaming { opacity: 0.75; }
  .stream-cursor {
    display: inline-block;
    width: 2px;
    height: 0.9em;
    background: var(--teal);
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: cur-blink 0.7s step-start infinite;
  }
  @keyframes cur-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

  /* ── Copy Buttons ─────────────────────────────────── */
  .copy-btn { background: none; border: none; color: #555; cursor: pointer; padding: 3px; border-radius: 4px; transition: color 0.15s, background 0.15s; display: inline-flex; align-items: center; justify-content: center; }
  .copy-btn:hover { color: #4DB8A4; background: rgba(77, 184, 164, 0.08); }
  .copy-btn.copied { color: #4DB8A4; }
  .exchange-copy { position: absolute; top: 4px; right: 0; opacity: 0; transition: opacity 0.15s; }
  .exchange:hover .exchange-copy { opacity: 1; }

  /* ── Retrieval Bar ────────────────────────────────── */
  .retrieval-bar {
    min-height: 44px;
    border-top: 1px solid var(--line-subtle);
    background: var(--deep);
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 28px;
    overflow-x: auto;
    flex-shrink: 0;
  }
  .retrieval-bar:empty { display: none; }
  .skill-tag {
    background: rgba(77, 184, 164, 0.1);
    border: 1px solid rgba(77, 184, 164, 0.2);
    color: #4DB8A4;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 4px;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .type-tag {
    background: rgba(232, 149, 106, 0.1);
    border: 1px solid rgba(232, 149, 106, 0.2);
    color: #E8956A;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 4px;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .retrieval-context {
    font-size: 11px;
    color: #666;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }

  /* ── Capture Bar ──────────────────────────────────── */
  .capture-bar {
    height: 40px;
    background: var(--deep);
    border-top: 1px solid #2a2a2a;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    flex-shrink: 0;
  }

  .cap-key {
    font-size: 11px;
    font-weight: 400;
    font-family: 'JetBrains Mono', monospace;
    color: #888;
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 2px 8px;
    letter-spacing: 0;
  }

  .cap-hint { font-size: 12px; color: #555; font-weight: 400; }

  /* ── Info Bar (past session) ──────────────────────── */
  .info-bar {
    height: 40px;
    background: var(--deep);
    border-top: 1px solid var(--line-subtle);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    flex-shrink: 0;
  }

  .info-bar-left {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 11px;
    color: var(--text-ghost);
    font-weight: 300;
  }

  .info-stat { display: flex; align-items: center; gap: 5px; }
  .info-stat strong { color: var(--text-dim); font-weight: 500; }
  .info-sep { color: var(--line); }

  .btn-resume {
    background: none;
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 4px 14px;
    font-size: 11px;
    font-weight: 500;
    font-family: 'DM Sans', 'Outfit', sans-serif;
    color: var(--text-dim);
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn-resume:hover { border-color: var(--pulse); color: var(--pulse); }

  /* ── Agent Panel ──────────────────────────────────── */
  .agent {
    width: 340px;
    background: var(--deep);
    border-left: 1px solid var(--line-subtle);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    position: relative;
  }

  .agent-resize-handle {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 6px;
    cursor: col-resize;
    z-index: 10;
    background: transparent;
    transition: background 0.15s;
  }
  .agent-resize-handle:hover { background: rgba(77, 184, 164, 0.15); }
  .agent-resize-handle.dragging { background: rgba(77, 184, 164, 0.3); }

  .agent-header {
    padding: 18px 20px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .agent-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #666;
  }

  .agent-mode {
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-ghost);
    background: var(--raised);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 2px 8px;
  }

  .agent-thread {
    flex: 1;
    overflow-y: auto;
    padding: 4px 16px 12px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .v-msg {
    max-width: 88%;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 13px;
    line-height: 1.55;
    font-weight: 400;
  }

  .v-msg.you {
    align-self: flex-end;
    background: var(--pulse-dim);
    color: var(--text-bright);
    border-bottom-right-radius: 4px;
    border: 1px solid rgba(212, 145, 90, 0.08);
  }

  .v-msg.twin {
    align-self: flex-start;
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--line);
    border-bottom-left-radius: 4px;
  }

  .v-msg.you.context, .v-msg.twin.context { opacity: 0.5; font-size: 12px; }
  .v-msg p { margin: 0 0 8px 0; }
  .v-msg p:last-child { margin-bottom: 0; }
  .v-msg.twin p { border-left: 2px solid rgba(77, 184, 164, 0.25); padding-left: 12px; }

  .context-divider {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0;
  }

  .context-divider-line { flex: 1; height: 1px; background: var(--line); }

  .context-divider-text {
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-ghost);
    white-space: nowrap;
  }

  .agent-compose {
    padding: 12px 16px 18px;
    border-top: 1px solid var(--line-subtle);
  }

  .compose-row { display: flex; gap: 8px; align-items: flex-end; }

  .compose-input {
    flex: 1;
    resize: none;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13px;
    font-family: 'DM Sans', 'Outfit', sans-serif;
    font-weight: 400;
    background: var(--surface);
    color: var(--text-bright);
    outline: none;
    min-height: 40px;
    max-height: 120px;
    transition: border-color 0.2s;
  }

  .compose-input::placeholder { color: var(--text-ghost); font-weight: 300; }
  .compose-input:focus { border-color: var(--pulse); box-shadow: 0 0 0 1px var(--pulse-dim); }

  .btn-speak {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    border: none;
    background: var(--teal);
    color: var(--void);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: all 0.15s;
    box-shadow: 0 2px 8px var(--teal-dim);
  }

  .btn-speak:hover { transform: translateY(-1px); box-shadow: 0 4px 16px var(--teal-medium); }
  .btn-speak svg { width: 15px; height: 15px; }

  /* ── Scrollbar ────────────────────────────────────── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #444; }

  /* ── End Session Modal ────────────────────────────── */
  .end-session-modal {
    position: fixed; inset: 0; z-index: 1000;
    display: flex; align-items: center; justify-content: center;
    background: rgba(0,0,0,0.6);
  }
  .end-session-card {
    background: #252525; border-radius: 8px; padding: 24px; max-width: 360px;
  }
  .end-session-title { font-size: 14px; color: #ddd; margin-bottom: 6px; font-weight: 500; }
  .end-session-subtitle { font-size: 13px; color: #888; margin-bottom: 20px; }
  .end-session-actions { display: flex; gap: 10px; justify-content: flex-end; }
  .end-session-btn-end {
    background: #E8956A; color: #1a1a1a; border: none; border-radius: 6px;
    padding: 8px 16px; font-size: 13px; font-weight: 500; cursor: pointer;
  }
  .end-session-btn-cancel {
    background: transparent; color: #888; border: 1px solid #888; border-radius: 6px;
    padding: 8px 16px; font-size: 13px; cursor: pointer;
  }
  .session-ended-label { font-size: 12px; color: #666; }
</style>
</head>
<body>

<!-- ═══ PRESENCE BAR ═══ -->
<div class="presence">
  <div class="identity">
    <button class="toggle-memory" id="toggleBtn" title="Toggle sessions">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
        <line x1="9" y1="3" x2="9" y2="21"/>
      </svg>
    </button>
    <div class="twin-mark" id="twinMark"></div>
    <div class="twin-name">Twin</div>
  </div>
  <div class="presence-center" id="presenceCenter">
    <button class="btn-session start" onclick="toggleMeeting()">Start Meeting</button>
  </div>
  <div class="capture-state inactive" id="captureState">
    <div class="capture-dot"></div>
    <span id="captureText">Inactive</span>
  </div>
</div>

<!-- ═══ THREE SPACES ═══ -->
<div class="spaces">

  <!-- Memory Lane -->
  <div class="memory" id="memoryPanel">
    <div class="memory-inner">
      <div class="memory-header">
        <span class="memory-label">Sessions</span>
        <button class="btn-new" onclick="createAndSwitchSession()" title="New session">+</button>
      </div>
      <div class="memory-list" id="sessionList">
        <div class="placeholder-text" style="padding: 8px 12px;">Loading...</div>
      </div>
    </div>
  </div>

  <!-- The Stage -->
  <div class="stage">
    <div class="stage-half" id="transcriptHalf">
      <div class="stage-label">Transcript <button class="copy-btn" onclick="copyTranscript(this)" aria-label="Copy transcript"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button></div>
      <div class="scroll-area" id="transcriptArea">
        <div class="placeholder-text">Transcript will appear here when the meeting starts...</div>
      </div>
    </div>

    <div class="divider" id="stageDivider"></div>

    <div class="stage-half" id="exchangeHalf" style="flex: 1.3;">
      <div class="stage-label">Exchange <button class="copy-btn" onclick="copyAllExchanges(this)" aria-label="Copy all exchanges"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button></div>
      <div class="scroll-area" id="exchangeArea">
        <div class="placeholder-text">Hold SPACE to capture a question...</div>
      </div>
    </div>

    <!-- Capture bar (live) -->
    <div class="capture-bar" id="captureBar">
      <span class="cap-key">SPACE</span>
      <span class="cap-hint">hold to capture</span>
    </div>

    <!-- Info bar (past sessions) -->
    <div class="info-bar" id="infoBar" style="display: none;">
      <div class="info-bar-left">
        <div class="info-stat"><strong id="infoExchanges">0</strong> exchanges</div>
        <span class="info-sep">·</span>
        <div class="info-stat" id="infoDate">—</div>
      </div>
      <button class="btn-resume" onclick="createAndSwitchSession()">New Session</button>
    </div>
  </div>

  <!-- The Agent -->
  <div class="agent" id="agentPanel">
    <div class="agent-resize-handle" id="agentResizeHandle"></div>
    <div class="agent-header">
      <span class="agent-label">Agent</span>
      <span class="agent-mode" id="agentMode" style="display: none;">Debrief</span>
    </div>
    <div class="agent-thread" id="agentThread">
      <div class="placeholder-text">Talk to your twin before, during, or after the interview...</div>
    </div>
    <div class="agent-compose" id="agentCompose">
      <div class="compose-row">
        <textarea class="compose-input" id="composeInput" placeholder="Talk to your twin..." rows="1"></textarea>
        <button class="btn-speak" onclick="sendAgentMessage()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
        </button>
      </div>
    </div>
  </div>

</div>

<div id="endSessionModal" class="end-session-modal" style="display: none;">
  <div class="end-session-card">
    <div class="end-session-title">End this session?</div>
    <div class="end-session-subtitle">This session will be locked. You won't be able to restart it.</div>
    <div class="end-session-actions">
      <button type="button" class="end-session-btn-cancel" onclick="hideEndSessionModal()">Cancel</button>
      <button type="button" class="end-session-btn-end" onclick="confirmEndSession()">End Session</button>
    </div>
  </div>
</div>

<script>
// ─── State ───────────────────────────────────────────────────────
let currentSessionId  = null;
let meetingActive     = false;
let isRecording       = false;
let viewingPastSession = false;
let sessionJustEnded  = false;
let sessions          = [];
let agentBusy         = false;

// ─── SSE ──────────────────────────────────────────────────────────
const es = new EventSource('/stream');
es.onmessage = (e) => {
  try { handleSSE(JSON.parse(e.data)); } catch (err) {}
};

function handleSSE(d) {
  switch (d.type) {
    case 'transcript':
      if (!viewingPastSession) appendTranscript(d.text);
      break;
    case 'exchange':
      if (!viewingPastSession) appendExchange(d);
      break;
    case 'exchange_start':
      if (!viewingPastSession) startStreamingExchange(d);
      break;
    case 'meeting_started':
      meetingActive = true;
      enterLiveMode(d.session_id);
      break;
    case 'meeting_stopped':
      meetingActive = false;
      if (d.session_id && d.session_id === currentSessionId) {
        enterEndedSessionMode();
        loadSessionIntoUI(currentSessionId);
      } else {
        enterInactiveMode();
      }
      refreshSessions();
      break;
    case 'recording_started':
      setStatus('capturing');
      break;
    case 'recording_stopped':
      setStatus('processing');
      break;
    case 'session_update':
      refreshSessions();
      break;
    case 'status':
      if (!viewingPastSession) setStatus(d.state);
      break;
    case 'error':
      if (!viewingPastSession) setStatus(meetingActive ? 'listening' : 'inactive');
      break;
  }
}

// ─── Init ─────────────────────────────────────────────────────────
async function init() {
  setupDivider();
  setupAgentResize();
  setupKeyboard();
  setupCompose();
  setupSidebarToggle();

  try {
    const [sessionsData, currentData] = await Promise.all([
      fetch('/api/sessions').then(r => r.json()),
      fetch('/api/current_session').then(r => r.json()),
    ]);

    sessions = sessionsData;
    renderSessionList();

    if (currentData.meeting_active && currentData.session_id) {
      meetingActive      = true;
      currentSessionId   = currentData.session_id;
      viewingPastSession = false;
      await loadSessionIntoUI(currentSessionId);
      enterLiveMode(currentSessionId);
    } else if (sessions.length > 0) {
      currentSessionId   = sessions[0].id;
      viewingPastSession = false;
      await loadSessionIntoUI(currentSessionId);
      setStatus('inactive');
      updatePresenceBar(false, false);
    } else {
      await createAndSwitchSession();
    }
  } catch (err) {
    console.error('Init error:', err);
  }
}

// ─── Sessions Sidebar ─────────────────────────────────────────────
async function refreshSessions() {
  try {
    sessions = await fetch('/api/sessions').then(r => r.json());
    renderSessionList();
  } catch (err) {}
}

function renderSessionList() {
  const list = document.getElementById('sessionList');
  if (!sessions.length) {
    list.innerHTML = '<div class="placeholder-text" style="padding:8px 12px;">No sessions yet</div>';
    return;
  }
  list.innerHTML = sessions.map(s => {
    let cls = '';
    if (s.id === currentSessionId) {
      cls = (meetingActive && s.is_meeting_active) ? 'active' : (sessionJustEnded || viewingPastSession ? 'viewing' : 'active');
    }
    const when = formatSessionDate(s.created_at);
    const sid  = esc(s.id);
    const name = esc(s.name);
    return `<div class="mem-item ${cls}" data-session-id="${sid}" onclick="handleMemItemClick(event, '${sid}')">
      <div class="mem-row">
        <div class="mem-name">${name}</div>
        <div class="mem-actions">
          <button class="mem-icon" title="Rename" onclick="startRename(event, '${sid}')">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
              <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
            </svg>
          </button>
          <button class="mem-icon delete" title="Delete" onclick="deleteSession(event, '${sid}')">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="3 6 5 6 21 6"/>
              <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
              <path d="M10 11v6M14 11v6"/>
              <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
            </svg>
          </button>
        </div>
      </div>
      <div class="mem-when">${esc(when)}</div>
    </div>`;
  }).join('');
}

function handleMemItemClick(e, sessionId) {
  // Don't navigate if clicking an action button or an active input
  if (e.target.closest('.mem-actions') || e.target.closest('.mem-name-input')) return;
  switchToSession(sessionId);
}

function startRename(e, sessionId) {
  e.stopPropagation();
  const item = document.querySelector(`.mem-item[data-session-id="${sessionId}"]`);
  if (!item) return;
  const nameEl = item.querySelector('.mem-name');
  const currentName = nameEl.textContent;

  // Replace name div with input
  const input = document.createElement('input');
  input.type      = 'text';
  input.className = 'mem-name-input';
  input.value     = currentName;
  nameEl.replaceWith(input);
  input.focus();
  input.select();

  const commit = async () => {
    const raw       = input.value.trim();
    const newName   = raw ? sanitizeName(raw) : currentName;
    const session   = sessions.find(s => s.id === sessionId);
    const oldFile   = session ? (session.id + '.json') : null;

    if (!oldFile || newName === currentName) {
      // Restore display without change
      const restored = document.createElement('div');
      restored.className = 'mem-name';
      restored.textContent = currentName;
      input.replaceWith(restored);
      return;
    }

    try {
      const res  = await fetch('/rename_session', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ session_id: sessionId, new_name: newName }),
      });
      const data = await res.json();
      if (data.ok) {
        // Update local sessions array
        const idx = sessions.findIndex(s => s.id === sessionId);
        if (idx !== -1) sessions[idx].name = newName;
        renderSessionList();
      } else {
        const restored = document.createElement('div');
        restored.className = 'mem-name';
        restored.textContent = currentName;
        input.replaceWith(restored);
      }
    } catch (err) {
      const restored = document.createElement('div');
      restored.className = 'mem-name';
      restored.textContent = currentName;
      input.replaceWith(restored);
    }
  };

  input.addEventListener('blur',    commit);
  input.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter')  { ev.preventDefault(); input.blur(); }
    if (ev.key === 'Escape') { input.value = currentName; input.blur(); }
  });
}

function sanitizeName(s) {
  // Allow letters, numbers, spaces, hyphens, underscores, dots, parens
  return s.replace(/[\/\\:*?"<>|]/g, '').trim().slice(0, 80) || 'Session';
}

async function deleteSession(e, sessionId) {
  e.stopPropagation();
  if (!confirm('Delete this session?')) return;

  try {
    const res  = await fetch('/delete_session', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ session_id: sessionId }),
    });
    const data = await res.json();
    if (data.ok) {
      sessions = sessions.filter(s => s.id !== sessionId);
      // If we deleted the current session, switch to the next one
      if (sessionId === currentSessionId) {
        if (sessions.length > 0) {
          await switchToSession(sessions[0].id);
        } else {
          currentSessionId   = null;
          viewingPastSession = false;
          clearStage();
          clearAgent();
          updatePresenceBar(false, false);
          setStatus('inactive');
        }
      }
      renderSessionList();
    }
  } catch (err) { console.error('Delete failed', err); }
}

async function createAndSwitchSession() {
  if (meetingActive) return;
  try {
    const session = await fetch('/api/sessions', { method: 'POST' }).then(r => r.json());
    sessions = await fetch('/api/sessions').then(r => r.json());
    currentSessionId   = session.id;
    viewingPastSession = false;
    sessionJustEnded   = false;
    document.getElementById('agentCompose').style.display = '';
    clearStage();
    clearAgent();
    setStatus('inactive');
    updatePresenceBar(false, false);
    renderSessionList();
  } catch (err) { console.error(err); }
}

async function switchToSession(sessionId) {
  if (meetingActive && sessionId !== currentSessionId) return;
  try {
    const sInfo = sessions.find(s => s.id === sessionId);
    const isLiveSession = meetingActive && sInfo && sInfo.is_meeting_active;
    currentSessionId   = sessionId;
    viewingPastSession = !isLiveSession;
    await loadSessionIntoUI(sessionId);
    renderSessionList();
    if (isLiveSession) {
      enterLiveMode(sessionId);
    } else {
      enterPastSessionMode(sessionId);
    }
  } catch (err) { console.error(err); }
}

async function loadSessionIntoUI(sessionId) {
  try {
    const data = await fetch(`/api/sessions/${sessionId}`).then(r => r.json());

    // Transcript
    const tArea = document.getElementById('transcriptArea');
    const transcript = typeof data.transcript === 'string'
      ? data.transcript
      : (Array.isArray(data.transcript)
          ? data.transcript.map(t => (typeof t === 'object' ? t.text : t)).join(' ')
          : '');
    if (transcript.trim()) {
      tArea.innerHTML = `<div class="transcript-line">${esc(transcript)}</div>`;
    } else {
      tArea.innerHTML = '<div class="placeholder-text">Transcript will appear here when the meeting starts...</div>';
    }

    // Exchanges
    const eArea = document.getElementById('exchangeArea');
    const exchanges = data.exchanges || [];
    const legacy    = data.qa_pairs  || [];

    if (exchanges.length > 0) {
      eArea.innerHTML = exchanges.map(ex => buildExchangeHTML(ex)).join('');
    } else if (legacy.length > 0) {
      eArea.innerHTML = legacy.map(qp => buildLegacyHTML(qp)).join('');
    } else {
      eArea.innerHTML = '<div class="placeholder-text">No exchanges in this session.</div>';
    }

    // Agent thread
    renderAgentThread(data);

    // Info bar stats
    const exCount = exchanges.length || legacy.length;
    document.getElementById('infoExchanges').textContent = exCount;
    const sessionTs = data.created_at || data.started_at || '';
    document.getElementById('infoDate').textContent = sessionTs ? formatSessionDate(sessionTs) : '—';

    // Store name/date for presence bar
    document.getElementById('presenceCenter').dataset.sessionName = data.name || 'Session';
    document.getElementById('presenceCenter').dataset.sessionDate = sessionTs ? formatSessionDate(sessionTs) : '';
  } catch (err) { console.error('loadSessionIntoUI error:', err); }
}

// ─── UI Modes ─────────────────────────────────────────────────────
function enterLiveMode(sessionId) {
  if (sessionId) currentSessionId = sessionId;
  viewingPastSession = false;
  sessionJustEnded   = false;
  document.getElementById('captureBar').style.display = '';
  document.getElementById('infoBar').style.display    = 'none';
  document.getElementById('agentMode').style.display  = 'none';
  document.getElementById('agentCompose').style.display = '';
  document.getElementById('twinMark').classList.add('breathing');
  updatePresenceBar(true, false);
  setStatus('listening');
  renderSessionList();
}

function enterInactiveMode() {
  viewingPastSession = false;
  sessionJustEnded   = false;
  document.getElementById('captureBar').style.display = '';
  document.getElementById('infoBar').style.display    = 'none';
  document.getElementById('agentMode').style.display  = 'none';
  document.getElementById('agentCompose').style.display = '';
  document.getElementById('twinMark').classList.remove('breathing');
  updatePresenceBar(false, false);
  setStatus('inactive');
  renderSessionList();
}

function enterPastSessionMode() {
  viewingPastSession = true;
  sessionJustEnded   = false;
  document.getElementById('captureBar').style.display = 'none';
  document.getElementById('infoBar').style.display    = '';
  document.getElementById('agentMode').style.display  = '';
  document.getElementById('agentCompose').style.display = '';
  document.getElementById('twinMark').classList.remove('breathing');
  updatePresenceBar(false, true);
  setStatus('inactive');
  renderSessionList();
}

function enterEndedSessionMode() {
  sessionJustEnded   = true;
  viewingPastSession = false;
  document.getElementById('captureBar').style.display = 'none';
  document.getElementById('infoBar').style.display    = '';
  document.getElementById('agentMode').style.display  = 'none';
  document.getElementById('agentCompose').style.display = 'none';
  document.getElementById('twinMark').classList.remove('breathing');
  updatePresenceBar(false, false);
  setStatus('inactive');
  renderSessionList();
}

function showEndSessionModal() {
  document.getElementById('endSessionModal').style.display = 'flex';
}

function hideEndSessionModal() {
  document.getElementById('endSessionModal').style.display = 'none';
}

async function confirmEndSession() {
  hideEndSessionModal();
  try {
    await fetch('/stop_meeting', { method: 'POST' });
    meetingActive = false;
    enterEndedSessionMode();
    await loadSessionIntoUI(currentSessionId);
    refreshSessions();
  } catch (err) { console.error(err); }
}

function updatePresenceBar(isLive, isPast) {
  const center = document.getElementById('presenceCenter');
  if (sessionJustEnded) {
    center.innerHTML = '<span class="session-ended-label">Session ended</span>';
  } else if (isLive) {
    center.innerHTML = '<button class="btn-session live" onclick="showEndSessionModal()">End Session</button>';
  } else if (isPast) {
    const name = center.dataset.sessionName || 'Past Session';
    const date = center.dataset.sessionDate || '';
    center.innerHTML = `<div class="session-viewing">
      <span class="badge-past">Past</span>
      <span class="session-viewing-name">${esc(name)}</span>
      <span class="session-viewing-date">${esc(date)}</span>
    </div>`;
  } else {
    center.innerHTML = '<button class="btn-session start" onclick="toggleMeeting()">Start Meeting</button>';
  }
}

function clearStage() {
  document.getElementById('transcriptArea').innerHTML =
    '<div class="placeholder-text">Transcript will appear here when the meeting starts...</div>';
  document.getElementById('exchangeArea').innerHTML =
    '<div class="placeholder-text">Hold SPACE to capture a question...</div>';
}

function clearAgent() {
  document.getElementById('agentThread').innerHTML =
    '<div class="placeholder-text">Talk to your twin before, during, or after the interview...</div>';
}

// ─── Copy Utilities ────────────────────────────────────────────────
function copyToClipboard(text, buttonEl) {
  if (!text.trim()) return;
  navigator.clipboard.writeText(text).then(() => {
    const original = buttonEl.innerHTML;
    buttonEl.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
    buttonEl.classList.add('copied');
    setTimeout(() => {
      buttonEl.innerHTML = original;
      buttonEl.classList.remove('copied');
    }, 1500);
  });
}

function copyTranscript(btn) {
  const line = document.querySelector('#transcriptArea .transcript-line');
  copyToClipboard(line ? line.textContent.trim() : '', btn);
}

function copyAllExchanges(btn) {
  const blocks = document.querySelectorAll('#exchangeArea .exchange');
  const lines = [];
  blocks.forEach(ex => {
    const speakerName = ex.querySelector('.who-name:not(.twin)')?.textContent || 'Speaker';
    const question    = ex.querySelector('.who-text')?.textContent?.trim() || '';
    const beats       = [...ex.querySelectorAll('.beat')].map(b => b.textContent.trim());
    lines.push(`${speakerName}: ${question}`);
    if (beats.length) lines.push(`Twin: ${beats.join(' ')}`);
    lines.push('');
  });
  copyToClipboard(lines.join('\\n').trim(), btn);
}

function copyExchange(btn) {
  const ex          = btn.closest('.exchange');
  const speakerName = ex.querySelector('.who-name:not(.twin)')?.textContent || 'Speaker';
  const question    = ex.querySelector('.who-text')?.textContent?.trim() || '';
  const beats       = [...ex.querySelectorAll('.beat')].map(b => b.textContent.trim());
  let text = `${speakerName}: ${question}`;
  if (beats.length) text += `\\nTwin: ${beats.join(' ')}`;
  copyToClipboard(text, btn);
}

// ─── Transcript ────────────────────────────────────────────────────
function appendTranscript(text) {
  const area = document.getElementById('transcriptArea');
  let line = area.querySelector('.transcript-line');
  if (!line) {
    area.innerHTML = '';
    line = document.createElement('div');
    line.className = 'transcript-line';
    area.appendChild(line);
  }
  line.textContent += (line.textContent ? ' ' : '') + text;
  area.scrollTop = area.scrollHeight;
}

// ─── Exchange Rendering ────────────────────────────────────────────
function buildExchangeHTML(ex) {
  const cls      = ex.speaker || 'interviewer';
  const name     = esc((ex.speaker_name || 'Speaker 1').toUpperCase());
  const text     = esc(ex.text || '');
  const beats    = (ex.beats || []).map(b => `<div class="beat">${esc(b)}</div>`).join('');
  return `<div class="exchange">
    <button class="copy-btn exchange-copy" onclick="copyExchange(this)" title="Copy exchange"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>
    <div class="who">
      <div class="who-dot ${cls}"></div>
      <span class="who-name ${cls}">${name}</span>
    </div>
    <div class="who-text">${text}</div>
    ${beats ? `<div class="who" style="margin-top:12px;">
      <div class="who-dot twin"></div>
      <span class="who-name twin">Twin</span>
    </div>
    <div class="beats">${beats}</div>` : ''}
  </div>`;
}

function buildLegacyHTML(qp) {
  const beats = (qp.response || '')
    .split('• ')
    .filter(b => b.trim())
    .map(b => `<div class="beat">${esc(b.trim())}</div>`)
    .join('');
  return `<div class="exchange">
    <button class="copy-btn exchange-copy" onclick="copyExchange(this)" title="Copy exchange"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>
    <div class="who">
      <div class="who-dot interviewer"></div>
      <span class="who-name interviewer">Speaker 1</span>
    </div>
    <div class="who-text">${esc(qp.question || '')}</div>
    ${beats ? `<div class="who" style="margin-top:12px;">
      <div class="who-dot twin"></div>
      <span class="who-name twin">Twin</span>
    </div>
    <div class="beats">${beats}</div>` : ''}
  </div>`;
}

function appendExchange(ex) {
  const area  = document.getElementById('exchangeArea');
  const empty = area.querySelector('.placeholder-text');
  if (empty) area.innerHTML = '';
  area.insertAdjacentHTML('afterbegin', buildExchangeHTML(ex));
  area.scrollTop = 0;
}

// ─── Streaming Exchange ────────────────────────────────────────────
function renderStreamingBeats(beatsEl, buffer) {
  const parts = buffer.split('\\u2022 ').map(p => p.trim()).filter(p => p.length > 0);
  if (!parts.length) return;
  let html = '';
  for (let i = 0; i < parts.length - 1; i++) {
    html += `<div class="beat">${esc(parts[i])}</div>`;
  }
  const last = parts[parts.length - 1];
  html += `<div class="beat beat-streaming">${esc(last)}<span class="stream-cursor"></span></div>`;
  beatsEl.innerHTML = html;
}

function startStreamingExchange(d) {
  const area = document.getElementById('exchangeArea');
  const empty = area.querySelector('.placeholder-text');
  if (empty) area.innerHTML = '';

  const cls  = esc(d.speaker || 'interviewer');
  const name = esc((d.speaker_name || 'Speaker 1').toUpperCase());
  const copyBtn = '<button class="copy-btn exchange-copy" onclick="copyExchange(this)" title="Copy exchange"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>';

  const html = `<div class="exchange" id="ex-${d.stream_id}">
    ${copyBtn}
    <div class="who">
      <div class="who-dot ${cls}"></div>
      <span class="who-name ${cls}">${name}</span>
    </div>
    <div class="who-text">${esc(d.text)}</div>
    <div class="who" style="margin-top:12px;">
      <div class="who-dot twin"></div>
      <span class="who-name twin">Twin</span>
    </div>
    <div class="beats" id="beats-${d.stream_id}"><div class="beat beat-streaming"><span class="stream-cursor"></span></div></div>
  </div>`;

  area.insertAdjacentHTML('afterbegin', html);
  area.scrollTop = 0;

  const beatsEl = document.getElementById('beats-' + d.stream_id);
  let buffer = '';

  const evtSrc = new EventSource('/api/stream_response?id=' + encodeURIComponent(d.stream_id));

  evtSrc.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'chunk') {
        buffer += msg.text;
        renderStreamingBeats(beatsEl, buffer);
      } else if (msg.type === 'done') {
        beatsEl.innerHTML = (msg.beats || []).map(b => `<div class="beat">${esc(b)}</div>`).join('');
        area.scrollTop = 0;
        evtSrc.close();
      } else if (msg.type === 'error') {
        beatsEl.innerHTML = `<div class="beat">${esc(msg.message)}</div>`;
        area.scrollTop = 0;
        evtSrc.close();
      }
    } catch (_) {}
  };

  evtSrc.onerror = () => {
    if (buffer.trim()) {
      const parts = buffer.split('\\u2022 ').map(p => p.trim()).filter(p => p);
      beatsEl.innerHTML = parts.map(b => `<div class="beat">${esc(b)}</div>`).join('');
    } else {
      beatsEl.innerHTML = '<div class="beat">Connection error — no response received.</div>';
    }
    evtSrc.close();
  };
}

// ─── Agent Thread ──────────────────────────────────────────────────
function formatMessage(text) {
  const escaped = esc(text);
  const paragraphs = escaped.split('\\n\\n').map(p => p.replace(/\\n/g, '<br>'));
  return '<p>' + paragraphs.join('</p><p>') + '</p>';
}

function renderAgentThread(sessionData) {
  const thread   = document.getElementById('agentThread');
  const messages = sessionData.agent_messages || [];

  if (!messages.length) {
    thread.innerHTML = '<div class="placeholder-text">Talk to your twin before, during, or after the interview...</div>';
    return;
  }

  if (!viewingPastSession) {
    thread.innerHTML = messages
      .map(m => `<div class="v-msg ${m.role === 'user' ? 'you' : 'twin'}">${formatMessage(m.content)}</div>`)
      .join('');
  } else {
    // Past session: dim pre-meeting messages, separate with divider
    const exCount = (sessionData.exchanges || []).length + (sessionData.qa_pairs || []).length;
    // Heuristic: if there are exchanges, messages before the last "start" keyword were pre-meeting
    const startIdx = findMeetingStartIndex(messages);
    let html = '';
    messages.forEach((m, i) => {
      const cls     = m.role === 'user' ? 'you' : 'twin';
      const ctxCls  = (startIdx > 0 && i < startIdx) ? ' context' : '';
      html += `<div class="v-msg ${cls}${ctxCls}">${formatMessage(m.content)}</div>`;
      if (i === startIdx - 1 && startIdx > 0) {
        html += `<div class="context-divider">
          <div class="context-divider-line"></div>
          <span class="context-divider-text">Post-session</span>
          <div class="context-divider-line"></div>
        </div>`;
      }
    });
    thread.innerHTML = html;
  }

  thread.scrollTop = thread.scrollHeight;
}

function findMeetingStartIndex(messages) {
  const startPhrases = ['session is live', 'meeting is starting', 'go time', 'good luck'];
  for (let i = 0; i < messages.length; i++) {
    const content = (messages[i].content || '').toLowerCase();
    if (startPhrases.some(p => content.includes(p))) {
      return i + 1;
    }
  }
  return 0;
}

function appendAgentMessage(role, content) {
  const thread = document.getElementById('agentThread');
  const placeholder = thread.querySelector('.placeholder-text');
  if (placeholder) thread.innerHTML = '';
  const div = document.createElement('div');
  div.className = `v-msg ${role}`;
  div.innerHTML = formatMessage(content);
  thread.appendChild(div);
  thread.scrollTop = thread.scrollHeight;
}

// ─── Agent Chat ────────────────────────────────────────────────────
async function sendAgentMessage() {
  if (agentBusy) return;
  const input = document.getElementById('composeInput');
  const text  = input.value.trim();
  if (!text) return;

  input.value = '';
  input.style.height = 'auto';
  appendAgentMessage('you', text);
  agentBusy = true;

  const thread   = document.getElementById('agentThread');
  const thinking = document.createElement('div');
  thinking.className        = 'v-msg twin';
  thinking.style.color      = 'var(--text-ghost)';
  thinking.style.fontStyle  = 'italic';
  thinking.textContent      = '...';
  thread.appendChild(thinking);
  thread.scrollTop = thread.scrollHeight;

  try {
    const res  = await fetch('/agent_chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ message: text, session_id: currentSessionId }),
    });
    const data = await res.json();
    thinking.remove();

    if (data.response) appendAgentMessage('twin', data.response);

    if (data.action === 'start_meeting' && !meetingActive) {
      await toggleMeeting();
    } else if (data.action === 'stop_meeting' && meetingActive) {
      await toggleMeeting();
    } else if (data.action === 'new_session') {
      await createAndSwitchSession();
    }
  } catch (err) {
    thinking.remove();
    appendAgentMessage('twin', 'Sorry, something went wrong.');
  }
  agentBusy = false;
}

// ─── Meeting Control ───────────────────────────────────────────────
async function toggleMeeting() {
  if (viewingPastSession) return;
  if (!meetingActive) {
    try {
      const res  = await fetch('/start_meeting', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ session_id: currentSessionId }),
      });
      const data = await res.json();
      if (data.active) {
        meetingActive    = true;
        currentSessionId = data.session_id;
        enterLiveMode(data.session_id);
      }
    } catch (err) { console.error(err); }
  } else {
    try {
      await fetch('/stop_meeting', { method: 'POST' });
      meetingActive = false;
      enterInactiveMode();
    } catch (err) { console.error(err); }
  }
}

// ─── Status ────────────────────────────────────────────────────────
function setStatus(state) {
  const el   = document.getElementById('captureState');
  const text = document.getElementById('captureText');
  el.className = 'capture-state ' + state;
  const labels = {
    listening:  'Listening',
    capturing:  'Capturing',
    processing: 'Processing',
    inactive:   'Inactive',
  };
  text.textContent = labels[state] || state;
}

// ─── Spacebar ──────────────────────────────────────────────────────
function setupKeyboard() {
  document.addEventListener('keydown', (e) => {
    if (e.code !== 'Space' || e.repeat) return;
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
    if (!meetingActive || isRecording || viewingPastSession) return;
    e.preventDefault();
    isRecording = true;
    fetch('/start_recording', { method: 'POST' });
  });

  document.addEventListener('keyup', (e) => {
    if (e.code !== 'Space' || !isRecording) return;
    e.preventDefault();
    isRecording = false;
    fetch('/stop_recording', { method: 'POST' });
  });
}

// ─── Resizable Divider ─────────────────────────────────────────────
function setupDivider() {
  const divider   = document.getElementById('stageDivider');
  const topHalf   = document.getElementById('transcriptHalf');
  const bottomHalf = document.getElementById('exchangeHalf');
  let dragging    = false;

  divider.addEventListener('mousedown', (e) => {
    dragging = true;
    e.preventDefault();
    document.body.style.cursor     = 'row-resize';
    document.body.style.userSelect = 'none';
  });

  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const stage  = topHalf.parentElement;
    const rect   = stage.getBoundingClientRect();
    const offset = e.clientY - rect.top;
    const totalH = rect.height - 40;
    const pct    = (offset / totalH) * 100;
    if (pct > 15 && pct < 80) {
      topHalf.style.flex   = 'none';
      topHalf.style.height = pct + '%';
      bottomHalf.style.flex = '1';
    }
  });

  document.addEventListener('mouseup', () => {
    if (dragging) {
      dragging = false;
      document.body.style.cursor     = '';
      document.body.style.userSelect = '';
    }
  });
}

// ─── Agent Panel Resize ────────────────────────────────────────────
let agentPanelWidth = 340;

function setupAgentResize() {
  const handle = document.getElementById('agentResizeHandle');
  const panel  = document.getElementById('agentPanel');
  let dragging = false;
  let startX, startW;

  handle.addEventListener('mousedown', (e) => {
    dragging = true;
    startX   = e.clientX;
    startW   = panel.getBoundingClientRect().width;
    handle.classList.add('dragging');
    document.body.style.cursor     = 'col-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const dx   = startX - e.clientX;
    const newW = Math.min(600, Math.max(250, startW + dx));
    agentPanelWidth   = newW;
    panel.style.width = newW + 'px';
  });

  document.addEventListener('mouseup', () => {
    if (dragging) {
      dragging = false;
      handle.classList.remove('dragging');
      document.body.style.cursor     = '';
      document.body.style.userSelect = '';
    }
  });
}

// ─── Compose Auto-resize ───────────────────────────────────────────
function setupCompose() {
  const input = document.getElementById('composeInput');
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendAgentMessage();
    }
  });
}

// ─── Sidebar Toggle ────────────────────────────────────────────────
function setupSidebarToggle() {
  document.getElementById('toggleBtn').addEventListener('click', () => {
    document.getElementById('memoryPanel').classList.toggle('collapsed');
    document.getElementById('toggleBtn').classList.toggle('is-collapsed');
  });
}

// ─── Utils ─────────────────────────────────────────────────────────
function esc(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatSessionDate(iso) {
  if (!iso) return '';
  try {
    const d   = new Date(iso);
    const now = new Date();
    const isToday = d.toDateString() === now.toDateString();
    if (isToday) {
      return 'Today, ' + d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    }
    return (
      d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) +
      ', ' +
      d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
    );
  } catch (e) { return iso; }
}

// ─── Boot ──────────────────────────────────────────────────────────
init();
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log_info("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    log_info("Whisper loaded.")

    log_info("Loading embedding model (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    log_info("Embedding model loaded.")

    if not os.path.exists(KB_CHUNKS_FILE) or not os.path.exists(KB_EMBEDDINGS_FILE):
        log_error("Local KB index not found.")
        log_error("Run `python3 sync_kb.py` first to build the index, then restart overlay.py.")
        sys.exit(1)

    log_info("Loading local KB index...")
    with open(KB_CHUNKS_FILE) as f:
        kb_chunks = json.load(f)
    kb_embeddings = np.load(KB_EMBEDDINGS_FILE)["embeddings"]
    log_info(
        f"KB loaded: {len(kb_chunks)} chunks "
        f"({sum(1 for c in kb_chunks if c['layer']=='L1')} L1, "
        f"{sum(1 for c in kb_chunks if c['layer']=='L2')} L2)."
    )

    log_info(f"Starting at http://localhost:{PORT}")
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False)
