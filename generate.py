"""
Response generation: context retrieval, beat navigation, streaming display.
"""

import re
import os
from typing import Optional

from dotenv import load_dotenv
import anthropic

load_dotenv()

_anthropic_client = None
_response_cache: dict = {}  # (question, beat_text_key) → generated response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt.md")


def _load_system_prompt() -> str:
    """Load the coaching system prompt from system_prompt.md at runtime."""
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise RuntimeError(
            f"system_prompt.md not found at {SYSTEM_PROMPT_PATH}. "
            "Create this file to configure the coaching persona."
        )


# ── Anthropic client ──────────────────────────────────────────────────────────
def _get_client():
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


# ── Text helpers ──────────────────────────────────────────────────────────────
def _clean_beat(beat_text: str) -> str:
    """Strip header, prefix, and probe lines; return the core narrative."""
    clean = []
    for line in beat_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.lower().startswith("probe:"):
            continue
        if "|" in line and len(line.split("|")) >= 2:
            continue
        clean.append(line)
    return " ".join(clean)


def _extract_fallback(beat_text: str) -> str:
    """Fallback: first substantive sentence from the beat."""
    for line in beat_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("probe:"):
            continue
        if "|" in line and len(line.split("|")) >= 2:
            continue
        if len(line) > 20:
            end = re.search(r'[.!?]', line)
            return line[:end.end()].strip() if end else line
    return ""


def _beat_label(beat_str: str) -> str:
    """Strip 'Beat N: ' prefix, return just the descriptive label."""
    return re.sub(r'^Beat \d+:\s*', '', beat_str).strip()


def _fit_reason(question: str, chunk: dict) -> str:
    """Return 1–2 tags that explain why this story matches the question."""
    tags = chunk.get("tags", [])
    q_lower = question.lower()
    matched = [t for t in tags if t.lower() in q_lower]
    if matched:
        return " · ".join(matched[:2])
    if tags:
        return " · ".join(tags[:2])
    return "semantic match"


# ── LLM generation ────────────────────────────────────────────────────────────
def generate_response(question: str, chunk: dict, all_beats: list, beat_index: int) -> str:
    """
    Call Claude with the full coaching persona to generate exactly what
    Heba should say out loud right now. Cached per (question, beat).
    Falls back to extracted text on API error.
    """
    idx = min(beat_index, len(all_beats) - 1)
    current_beat = all_beats[idx]
    beat_text = current_beat.get("text", "")

    cache_key = (question[:80], beat_text[:120])
    if cache_key in _response_cache:
        return _response_cache[cache_key]

    # Build full story context: current beat + remaining beats as notes
    story_context = _clean_beat(beat_text)
    remaining = [_clean_beat(b.get("text", "")) for b in all_beats[idx + 1:]]
    if remaining:
        story_context += "\n\nRemainder of story (later beats):\n" + "\n".join(
            f"- {_beat_label(all_beats[idx + 1 + i].get('beat', ''))}: {t[:120]}"
            for i, t in enumerate(remaining)
        )

    user_message = (
        f"Interview question: \"{question}\"\n\n"
        f"Story: {chunk.get('company', '')} — {chunk.get('story', '')}\n"
        f"Current beat ({beat_index + 1}/{len(all_beats)}): "
        f"{_beat_label(current_beat.get('beat', ''))}\n\n"
        f"Notes:\n{story_context}"
    )

    try:
        client = _get_client()
        message = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            system=_load_system_prompt(),
            messages=[{"role": "user", "content": user_message}],
        )
        result = message.content[0].text.strip()
        _response_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"  (LLM error: {e} — using extracted text)")
        return _extract_fallback(beat_text)


# ── Question detection ────────────────────────────────────────────────────────
QUESTION_STARTERS = (
    "what", "how", "why", "when", "where", "who", "which",
    "can you", "could you", "tell me", "describe", "explain",
    "walk me through", "have you", "did you", "do you",
    "would you", "was there", "talk me through", "give me",
)


def is_question(text: str) -> bool:
    """Return True if the transcription looks like a real interview question."""
    text = text.strip()
    if len(text) < 10:
        return False
    if text.endswith("?"):
        return True
    lower = text.lower()
    if any(lower.startswith(s) for s in QUESTION_STARTERS):
        return True
    return False


def is_followup(text: str, current_story: Optional[dict]) -> bool:
    """
    Return True if this question continues the current story thread
    rather than opening a new topic.
    """
    if not current_story:
        return False

    text_lower = text.lower()

    # Very short clarifying questions are almost always follow-ups
    if len(text.split()) <= 7:
        return True

    # Tag overlap
    for tag in current_story.get("tags", []):
        if tag.lower() in text_lower:
            return True

    # Company or story name overlap
    if current_story.get("company", "").lower() in text_lower:
        return True
    if current_story.get("story", "").lower() in text_lower:
        return True

    return False


# ── Display ───────────────────────────────────────────────────────────────────
def format_display(question: str, chunk: dict, all_beats: list, beat_index: int) -> str:
    """
    Render the glanceable output:
      1. Cue line  — story + why it fits + beat position
      2. Response  — exactly what to say out loud (3–4 sentences)
      3. Beat nav  — what's next
    """
    company = chunk.get("company", "")
    story = chunk.get("story", "")
    total = len(all_beats)
    idx = min(beat_index, total - 1)
    current_beat = all_beats[idx]

    fit = _fit_reason(question, chunk)
    beat_title = _beat_label(current_beat.get("beat", ""))
    beat_num = idx + 1

    response = generate_response(question, chunk, all_beats, beat_index)

    sep = "─" * 52

    lines = [
        "",
        sep,
        f"  {company.upper()}: {story}",
        f"  ● {beat_num}/{total}  {beat_title}  [{fit}]",
        sep,
        "",
    ]

    # One sentence per line, blank line between each
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response) if s.strip()]
    for sentence in sentences:
        lines.append(f"  {sentence}")
        lines.append("")

    if idx + 1 < total:
        next_label = _beat_label(all_beats[idx + 1].get("beat", ""))
        lines.append(f"  ↓ next: {next_label}")

    lines += [sep, ""]

    return "\n".join(lines)

