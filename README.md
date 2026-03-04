# Twin

Real-time context engine. Listens to a live conversation, retrieves from a structured personal knowledge base, and streams relevant context in under a second.

## Why

Anyone who's been in a high-stakes conversation knows the problem: you have the information, you wrote it down, but you can't get to it without breaking the flow. You're either present or you're searching your notes. You can't do both.

Twin sits between your knowledge base and the live conversation. It transcribes ambient audio continuously, and when you need context on something being discussed, a spacebar press triggers retrieval against a structured KB. The matched content streams back beat-by-beat so the first line is visible in about a second.

## How It Works

Three-panel local app:

- **Transcript**: continuous ambient transcription via Whisper. Full conversation captured in real time.
- **Exchange**: spacebar triggers retrieval. The captured moment and matched KB context appear as a paired exchange.
- **Agent**: side channel to your own KB. Prep before, quick lookups during, debrief after.

```text
Ambient audio ──→ Whisper (local) ──→ Transcript panel
                                         ↓
Spacebar capture ──→ Whisper ──→ Query
                                    ↓
                        sentence-transformers (all-MiniLM-L6-v2)
                                    ↓
                        Cosine similarity vs. KB embeddings
                                    ↓
                        Top-k chunks → Claude API (streaming)
                                    ↓
                        Beat-by-beat response → Exchange panel
```

No cloud dependencies except Claude's API for response generation. Transcription, embeddings, and search all run locally.

## The Knowledge Base

This is where most of the work went. The KB architecture matters more than the pipeline.

**The problem with flat documents:** Dump a 3,000-word doc into a vector store because one paragraph is relevant and the model has to figure out which parts matter. It blends topics, pulls wrong details, and hallucinates to fill gaps. I'd already seen this pattern at scale. Most RAG accuracy failures trace to retrieval and chunking, not the model.

**The fix: structured knowledge units.** Each unit covers one topic. Each unit has metadata: company, type, skills, and a "When to Use" field that explicitly declares what should trigger retrieval. Units are broken into beats (`## Beat N`), and each beat is chunked independently. Retrieval pulls the specific beat, not the whole document.

Two layers:
- **Layer 1** (182 chunks): Knowledge units. Structured content broken into retrievable beats.
- **Layer 2** (207 chunks): Reasoning frameworks. Structural patterns for different conversational contexts.

389 chunks total. Full re-sync from Notion takes about 30 seconds.

**Why Notion as source of truth:** Content changes shouldn't require code changes. Edit a page, add a beat, change a trigger, re-sync. Same principle as config-driven systems.

**Why cosine similarity over a vector DB:** 389 chunks. NumPy is fast enough and adds zero infrastructure. At 10K+ chunks this decision changes.

**Why local Whisper over cloud transcription:** Spacebar capture needs low latency. You're waiting on the response while someone is talking. Network round-trips on every capture aren't acceptable.

## The Feedback Loop

A `tweaks.md` file is injected into the system prompt. One-line editorial rules added after each session. This separates two failure modes:

- Wrong chunks retrieved → fix the KB (adjust "When to Use" fields, re-sync)
- Right chunks, wrong emphasis → add a tweak (editorial correction in the prompt)

The system improves through encoded judgment, not retraining.

## What's Next

**Proactive nudges.** The transcript already runs continuously. Next version monitors the conversation and surfaces context without a manual trigger, catching things you haven't mentioned yet rather than responding to things you asked about.

## Stack

| Component | Tool | Why |
|-----------|------|-----|
| Transcription | Whisper (local) | No latency, no cloud dependency |
| Embeddings | all-MiniLM-L6-v2 | Fast, local, sufficient for about 400 chunks |
| Vector search | NumPy cosine similarity | No DB overhead at this scale |
| Response generation | Claude API (streaming) | Best quality for structured contextual responses |
| Knowledge base | Notion (2 databases) | Editable, structured, syncs in 30 seconds |
| Frontend | Flask + vanilla JS | Single-user local app, no framework needed |
| Audio | PyAudio + spacebar capture | Simple, reliable |
| IDE | Cursor | AI-assisted development |

## Files

```
├── overlay.py          # Flask app, three-panel UI, streaming responses
├── generate.py         # Claude API integration, prompt construction
├── audio_capture.py    # PyAudio recording, spacebar trigger
├── sync_kb.py          # Notion sync + embedding generation
├── tweaks.md           # Editorial corrections (injected into prompt)
└── pyproject.toml      # Dependencies
```

## Setup

```bash
git clone https://github.com/hebaasmar/twin.git
cd twin

echo "NOTION_TOKEN=your_token" > .env
echo "ANTHROPIC_API_KEY=your_key" >> .env

pip install -r requirements.txt  # or: uv sync

python3 sync_kb.py    # sync KB from Notion, generate embeddings
python3 overlay.py    # http://localhost:5055
```

## Full Case Study

Detailed technical writeup with architecture decisions, KB design rationale, and what I learned: [hebaasmar.notion.site](https://hebaasmar.notion.site)
