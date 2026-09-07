# -*- coding: utf-8 -*-
"""Hybrid RAG Retrieval Engine (Module 4 — Retrieval Component).

Implements FR-4.1 and FR-4.2 from the AgriDoc-PK PRD:
  - ChromaDB vector similarity search over PARC corpus
  - Disease-filtered semantic retrieval (Top-K)
  - Context assembly combining vision metadata + retrieved passages
  - Keyword boosting for exact chemical name matching
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COLLECTION_NAME = "parc_agronomy"
CHROMA_PERSIST_DIR = Path(__file__).resolve().parent.parent / "data_store" / "chroma_db"
DEFAULT_TOP_K = 3


class RAGEngine:
    """Hybrid RAG retrieval engine over the PARC agronomic corpus.

    Provides semantic search with disease-specific filtering and
    context assembly for the grounded LLM prompt.

    Args:
        persist_dir: Path to ChromaDB persistent storage.
        collection_name: Name of the ChromaDB collection.
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = COLLECTION_NAME,
    ):
        persist_path = Path(persist_dir) if persist_dir else CHROMA_PERSIST_DIR

        if not persist_path.exists():
            logger.warning(
                "ChromaDB directory not found at %s. "
                "Run 'python knowledge_base/build_index.py' first.",
                persist_path,
            )

        self._client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False),
        )

        try:
            self._collection = self._client.get_collection(collection_name)
            logger.info(
                "RAG engine initialized: collection='%s' (%d documents)",
                collection_name, self._collection.count(),
            )
        except Exception as e:
            logger.warning(
                "Collection '%s' not found: %s. "
                "Run 'python knowledge_base/build_index.py' to create it.",
                collection_name, e,
            )
            self._collection = None

    @property
    def is_ready(self) -> bool:
        """Check if the RAG engine has a populated index."""
        return self._collection is not None and self._collection.count() > 0

    def retrieve(
        self,
        query: str,
        disease_id: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K relevant passages from the PARC corpus.

        Performs semantic similarity search with optional disease-tag
        filtering for precision. Results are enriched with metadata
        for citation tracking.

        Args:
            query: Natural language query combining disease + severity context.
            disease_id: Optional disease identifier to filter results
                       (e.g., 'wheat_leaf_rust').
            top_k: Number of results to return (default 3).

        Returns:
            List of dicts with keys:
              - content (str): Retrieved text passage
              - source_file (str): Source document filename
              - section_heading (str): Section heading from document
              - distance (float): Similarity distance (lower = better)
              - relevance_score (float): Normalized relevance (0-1)
        """
        if not self.is_ready:
            logger.warning("RAG engine not ready — returning empty results")
            return []

        # Build where filter for disease-specific retrieval
        where_filter = None
        if disease_id:
            where_filter = {
                "disease_tags": {"$eq": disease_id}
            }

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )
        except Exception as e:
            logger.error("ChromaDB query failed: %s. Retrying without filter.", e)
            # Fallback: query without disease filter
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
            )

        # Parse results into structured format
        passages = []
        if results and results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)

            for doc, meta, dist in zip(documents, metadatas, distances):
                # Normalize distance to relevance score (1.0 = perfect match)
                relevance = max(0.0, 1.0 - (dist / 2.0))
                passages.append({
                    "content": doc,
                    "source_file": meta.get("source_file", "unknown"),
                    "section_heading": meta.get("section_heading", ""),
                    "distance": round(dist, 4),
                    "relevance_score": round(relevance, 4),
                })

        logger.info(
            "Retrieved %d passages for query='%s' (disease=%s)",
            len(passages), query[:80], disease_id,
        )
        return passages

    def build_context(
        self,
        disease_id: str,
        disease_name: str,
        severity_pct: float,
        severity_level: str,
        confidence: float,
        crop: str,
    ) -> Dict[str, Any]:
        """Assemble the full RAG context for the LLM prompt.

        Combines vision model outputs with retrieved PARC passages
        into a structured context dictionary (FR-4.2).

        Args:
            disease_id: Classified disease identifier.
            disease_name: Human-readable disease name.
            severity_pct: Infection percentage from severity engine.
            severity_level: Triage level (Mild/Moderate/Severe).
            confidence: Classification confidence (0.0-1.0).
            crop: Crop name (Wheat/Cotton/Rice).

        Returns:
            Context dictionary with vision_metadata, retrieved_passages,
            and formatted query string.
        """
        # Construct semantic query from vision outputs
        query = (
            f"{crop} {disease_name} treatment recommendation. "
            f"Severity: {severity_pct}% infection ({severity_level}). "
            f"Approved fungicide insecticide dosage application guidelines. "
            f"PARC official advisory Pakistan."
        )

        # Retrieve relevant passages
        passages = self.retrieve(query, disease_id=disease_id)

        # Extract citations
        citations = list({
            p["source_file"].replace(".md", "").replace("_", " ").title()
            for p in passages
        })

        context = {
            "vision_metadata": {
                "crop": crop,
                "disease_id": disease_id,
                "disease_name": disease_name,
                "confidence": confidence,
                "severity_pct": severity_pct,
                "severity_level": severity_level,
            },
            "retrieved_passages": passages,
            "citations": citations,
            "query_used": query,
        }

        logger.info(
            "Context assembled: disease=%s, severity=%.1f%%, passages=%d",
            disease_id, severity_pct, len(passages),
        )
        return context
