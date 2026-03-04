"""
sync_kb.py — Build the local knowledge base index for the interview coach.

Run this BEFORE an interview to pre-sync content from Notion.
During an interview, overlay.py does zero Notion API calls — only local search + Claude.

Usage:
    python3 sync_kb.py

Output:
    kb_chunks.json             — chunk text + metadata
    kb_chunk_embeddings.npz    — numpy array of embeddings (one row per chunk)
"""

import json
import os
import re
import sys

import httpx
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
if not NOTION_API_KEY:
    print("ERROR: NOTION_API_KEY not set in .env")
    sys.exit(1)

NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

# Notion database IDs
L1_DB_ID = "1db39014ff1848d0bf824c490f9b36fc"  # Knowledge Units (stories, facts)
L2_DB_ID = "8d97d7a08a4f4e9083303e1b5fa651f8"  # Reasoning Protocols

MAX_SECTION_CHARS = 600

BASE_DIR = os.path.expanduser("~/second-brain-dev")
CHUNKS_FILE = os.path.join(BASE_DIR, "kb_chunks.json")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "kb_chunk_embeddings.npz")


# ── Notion helpers ────────────────────────────────────────────────────────────
def notion_get(url: str) -> dict:
    resp = httpx.get(url, headers=NOTION_HEADERS, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def notion_post(url: str, body: dict) -> dict:
    resp = httpx.post(url, headers=NOTION_HEADERS, json=body, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def query_database(db_id: str) -> list:
    """Return all pages from a Notion database."""
    pages = []
    cursor = None
    while True:
        body = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor
        data = notion_post(f"https://api.notion.com/v1/databases/{db_id}/query", body)
        pages.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    return pages


def extract_rich_text(rt_array: list) -> str:
    return "".join(t.get("plain_text", "") for t in rt_array)


def extract_select(prop: dict) -> str:
    sel = prop.get("select")
    return sel.get("name", "") if sel else ""


def extract_multi_select(prop: dict) -> list:
    return [s.get("name", "") for s in prop.get("multi_select", [])]


def extract_title(prop: dict) -> str:
    return extract_rich_text(prop.get("title", []))


def extract_page_properties(page: dict, layer: str) -> dict:
    """Pull the fields we care about from a page's properties."""
    props = page.get("properties", {})
    title = ""
    for key in ("Name", "Title", "title", "Unit", "Protocol"):
        if key in props and props[key].get("type") == "title":
            title = extract_title(props[key])
            break

    if layer == "L1":
        tags = extract_multi_select(props.get("Skills Demonstrated", {}))
        when_to_use = extract_rich_text(
            props.get("When to Use", {}).get("rich_text", [])
        )
        page_type = extract_select(props.get("Type", {}))
        question_patterns = ""
    else:
        tags = extract_multi_select(props.get("Layer 2 Tags to Pull", {}))
        when_to_use = ""
        question_patterns = extract_rich_text(
            props.get("Question Patterns", {}).get("rich_text", [])
        )
        page_type = extract_select(props.get("Type", {}))

    return {
        "title": title,
        "page_id": page["id"],
        "layer": layer,
        "type": page_type,
        "tags": tags,
        "when_to_use": when_to_use or question_patterns,
    }


def fetch_page_blocks(page_id: str) -> list:
    """Fetch all blocks from a page (paginated)."""
    blocks = []
    cursor = None
    while True:
        url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        if cursor:
            url += f"?start_cursor={cursor}"
        data = notion_get(url)
        blocks.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    return blocks


def blocks_to_text(blocks: list) -> str:
    """Convert Notion blocks to plain text, preserving ## headings."""
    parts = []
    for block in blocks:
        btype = block.get("type", "")
        content = block.get(btype, {})

        if "rich_text" in content:
            text = extract_rich_text(content["rich_text"])
            if text.strip():
                if btype in ("heading_1", "heading_2", "heading_3"):
                    parts.append(f"\n## {text}\n")
                elif btype == "bulleted_list_item":
                    parts.append(f"- {text}")
                elif btype == "numbered_list_item":
                    parts.append(f"- {text}")
                else:
                    parts.append(text)

    return "\n".join(parts)


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_page(page_meta: dict, full_text: str) -> list:
    """
    Split a page into chunks:
    1. Split on ## heading boundaries first.
    2. If a section exceeds MAX_SECTION_CHARS, split further on paragraph boundaries.
    3. Prepend "When to Use" / "Question Patterns" to the first chunk for better embedding relevance.
    """
    when_to_use = page_meta.get("when_to_use", "").strip()

    # Split on ## headings
    sections = re.split(r"(?=\n## )", full_text)
    sections = [s.strip() for s in sections if s.strip()]

    if not sections:
        sections = [full_text.strip()] if full_text.strip() else []

    raw_chunks = []
    for section in sections:
        if len(section) <= MAX_SECTION_CHARS:
            raw_chunks.append(section)
        else:
            # Split on paragraph boundaries
            paragraphs = re.split(r"\n\n+", section)
            current = ""
            for para in paragraphs:
                if not para.strip():
                    continue
                if current and len(current) + len(para) + 2 > MAX_SECTION_CHARS:
                    raw_chunks.append(current.strip())
                    current = para
                else:
                    current = (current + "\n\n" + para).strip() if current else para
            if current.strip():
                raw_chunks.append(current.strip())

    chunks = []
    for i, text in enumerate(raw_chunks):
        if not text:
            continue
        # Prepend when_to_use to the first chunk so embedding matches question patterns
        if i == 0 and when_to_use:
            embed_text = f"When to use: {when_to_use}\n\n{text}"
        else:
            embed_text = text

        chunks.append({
            "page_title": page_meta["title"],
            "page_id":    page_meta["page_id"],
            "layer":      page_meta["layer"],
            "type":       page_meta["type"],
            "tags":       page_meta["tags"],
            "when_to_use": page_meta["when_to_use"],
            "text":       text,
            "embed_text": embed_text,
        })

    return chunks


# ── Main sync ─────────────────────────────────────────────────────────────────
def sync():
    all_chunks = []

    for layer, db_id in [("L1", L1_DB_ID), ("L2", L2_DB_ID)]:
        print(f"\nFetching {layer} pages from database {db_id}...")
        pages = query_database(db_id)
        print(f"  Found {len(pages)} pages in {layer}.")

        layer_chunks = []
        for page in pages:
            meta = extract_page_properties(page, layer)
            if not meta["title"]:
                continue

            print(f"  Processing [{layer}]: {meta['title']}")
            blocks = fetch_page_blocks(page["id"])
            print(f"    Blocks: {len(blocks)}, first block type: {blocks[0].get('type') if blocks else 'NONE'}")
            full_text = blocks_to_text(blocks)
            print(f"    Text length: {len(full_text)}")

            if not full_text.strip():
                print(f"    (empty — skipping)")
                continue

            chunks = chunk_page(meta, full_text)
            layer_chunks.extend(chunks)
            print(f"    → {len(chunks)} chunk(s)")

        print(f"  {layer} total: {len(layer_chunks)} chunks from {len(pages)} pages.")
        all_chunks.extend(layer_chunks)

    if not all_chunks:
        print("\nERROR: No chunks produced. Check Notion API key and database IDs.")
        sys.exit(1)

    print(f"\nEmbedding {len(all_chunks)} chunks with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_texts = [c["embed_text"] for c in all_chunks]
    embeddings = model.encode(embed_texts, normalize_embeddings=True, show_progress_bar=True)

    # Strip embed_text before saving (not needed at runtime)
    for c in all_chunks:
        del c["embed_text"]

    with open(CHUNKS_FILE, "w") as f:
        json.dump(all_chunks, f, indent=2)
    np.savez(EMBEDDINGS_FILE, embeddings=embeddings)

    l1_count = sum(1 for c in all_chunks if c["layer"] == "L1")
    l2_count = sum(1 for c in all_chunks if c["layer"] == "L2")
    l1_pages = len({c["page_id"] for c in all_chunks if c["layer"] == "L1"})
    l2_pages = len({c["page_id"] for c in all_chunks if c["layer"] == "L2"})

    print(f"\nSynced {l1_pages} pages, {l1_count} chunks from Layer 1.")
    print(f"Synced {l2_pages} pages, {l2_count} chunks from Layer 2.")
    print(f"Total embeddings: {len(all_chunks)}.")
    print(f"\nSaved:\n  {CHUNKS_FILE}\n  {EMBEDDINGS_FILE}")
    print("\nReady. Run `python3 overlay.py` to start the interview coach.")


if __name__ == "__main__":
    sync()
