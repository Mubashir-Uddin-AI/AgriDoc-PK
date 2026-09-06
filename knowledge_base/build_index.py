# -*- coding: utf-8 -*-
"""PARC Agronomic Corpus Indexer for ChromaDB.

Chunks the PARC advisory markdown documents into semantic segments
and indexes them in a ChromaDB persistent collection for RAG retrieval.

Chunking strategy (from PRD FR-4.1):
  - ~350 tokens per chunk with 50-token overlap
  - Approximate using character count (1 token ≈ 4 characters)
  - Metadata includes source document, section, and disease tags

Usage:
    python -m knowledge_base.build_index
    # or
    python knowledge_base/build_index.py
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE_CHARS = 1400        # ~350 tokens × 4 chars/token
OVERLAP_CHARS = 200            # ~50 tokens × 4 chars/token
COLLECTION_NAME = "parc_agronomy"

PARC_DOCS_DIR = Path(__file__).resolve().parent / "parc_documents"
CHROMA_PERSIST_DIR = Path(__file__).resolve().parent.parent / "data_store" / "chroma_db"

# Disease tag mapping by filename
DISEASE_TAG_MAP: Dict[str, List[str]] = {
    "wheat_rust_bulletin.md": ["wheat_leaf_rust", "wheat_stripe_rust"],
    "cotton_clcuv_guide.md": ["cotton_leaf_curl_virus"],
    "rice_blast_management.md": ["rice_blast", "rice_brown_spot"],
}


def _extract_sections(text: str) -> List[Dict[str, str]]:
    """Split a markdown document into logical sections by headings.

    Args:
        text: Full markdown document text.

    Returns:
        List of dicts with 'heading' and 'content' keys.
    """
    sections = []
    current_heading = "Introduction"
    current_lines: list[str] = []

    for line in text.split("\n"):
        heading_match = re.match(r"^#{1,4}\s+(.+)$", line)
        if heading_match:
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append({
                        "heading": current_heading,
                        "content": content,
                    })
            current_heading = heading_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Last section
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append({
                "heading": current_heading,
                "content": content,
            })

    return sections


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS,
                overlap: int = OVERLAP_CHARS) -> List[str]:
    """Split text into overlapping chunks at sentence boundaries.

    Tries to break at sentence endings (. or newline) to maintain
    semantic coherence within each chunk.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try to break at a sentence boundary
            break_point = text.rfind(". ", start, end)
            if break_point == -1:
                break_point = text.rfind("\n", start, end)
            if break_point > start:
                end = break_point + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


def build_index(
    docs_dir: str | Path | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Build the ChromaDB vector index from PARC markdown documents.

    Reads all .md files from the documents directory, chunks them
    with overlap, and upserts into a persistent ChromaDB collection
    with metadata tags for disease-specific retrieval.

    Args:
        docs_dir: Path to PARC markdown documents directory.
        persist_dir: Path for ChromaDB persistent storage.
        collection_name: Name of the ChromaDB collection.

    Returns:
        The populated ChromaDB Collection object.
    """
    docs_dir = Path(docs_dir) if docs_dir else PARC_DOCS_DIR
    persist_dir = Path(persist_dir) if persist_dir else CHROMA_PERSIST_DIR

    # Ensure persist directory exists
    persist_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building PARC index from %s → %s", docs_dir, persist_dir)

    # Initialize ChromaDB client with persistence
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete existing collection if it exists (rebuild from scratch)
    try:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection '%s'", collection_name)
    except ValueError:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "PARC Agronomic Advisory Corpus for AgriDoc-PK RAG"},
    )

    all_ids = []
    all_documents = []
    all_metadatas = []

    # Process each markdown document
    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        logger.warning("No markdown files found in %s", docs_dir)
        return collection

    for md_file in md_files:
        filename = md_file.name
        text = md_file.read_text(encoding="utf-8")
        disease_tags = DISEASE_TAG_MAP.get(filename, ["general"])

        logger.info("Processing %s (%d chars, tags=%s)",
                     filename, len(text), disease_tags)

        # Extract sections and chunk
        sections = _extract_sections(text)

        chunk_idx = 0
        for section in sections:
            heading = section["heading"]
            content = section["content"]

            # Prepend heading to content for better retrieval context
            full_text = f"{heading}\n\n{content}"
            chunks = _chunk_text(full_text)

            for chunk in chunks:
                doc_id = f"{filename}::section-{heading[:50]}::chunk-{chunk_idx}"
                metadata = {
                    "source_file": filename,
                    "section_heading": heading,
                    "disease_tags": ",".join(disease_tags),
                    "chunk_index": chunk_idx,
                    "source_organization": "PARC",
                }

                all_ids.append(doc_id)
                all_documents.append(chunk)
                all_metadatas.append(metadata)
                chunk_idx += 1

    # Batch upsert all chunks
    if all_documents:
        collection.add(
            ids=all_ids,
            documents=all_documents,
            metadatas=all_metadatas,
        )
        logger.info(
            "Indexed %d chunks from %d documents into collection '%s'",
            len(all_documents), len(md_files), collection_name,
        )
    else:
        logger.warning("No chunks generated — collection is empty")

    return collection


def main():
    """CLI entry point for building the PARC index."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    collection = build_index()
    print(f"\n[OK] PARC index built successfully: {collection.count()} chunks indexed")
    print(f"     Collection: {COLLECTION_NAME}")
    print(f"     Persist dir: {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()
