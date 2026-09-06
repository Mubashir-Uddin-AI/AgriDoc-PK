# -*- coding: utf-8 -*-
"""Edge-TTS Urdu Voice Synthesis Engine (Module 5).

Implements FR-5.1 through FR-5.3 from the AgriDoc-PK PRD:
  - High-fidelity Urdu speech via edge-tts neural voices
  - Support for ur-PK-AsadNeural (male) and ur-PK-UzmaNeural (female)
  - Async MP3 generation with synchronous wrapper for Streamlit
  - In-memory byte streaming (no temp files)
"""

from __future__ import annotations

import asyncio
import io
import logging
import hashlib
from typing import Optional

import edge_tts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice configuration
# ---------------------------------------------------------------------------
VOICE_MALE = "ur-PK-AsadNeural"
VOICE_FEMALE = "ur-PK-UzmaNeural"
DEFAULT_VOICE = VOICE_MALE

# Speech parameters
DEFAULT_RATE = "+0%"     # Normal speaking rate
DEFAULT_VOLUME = "+0%"   # Normal volume
DEFAULT_PITCH = "+0Hz"   # Normal pitch

# Cache for generated audio (keyed by text hash)
_audio_cache: dict[str, bytes] = {}


async def _synthesize_async(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    volume: str = DEFAULT_VOLUME,
    pitch: str = DEFAULT_PITCH,
) -> bytes:
    """Generate MP3 audio bytes from Urdu text using edge-tts.

    Args:
        text: Urdu text string to synthesize.
        voice: Edge-TTS voice identifier.
        rate: Speaking rate adjustment (e.g., "+10%", "-5%").
        volume: Volume adjustment.
        pitch: Pitch adjustment.

    Returns:
        MP3 audio bytes.

    Raises:
        RuntimeError: If TTS synthesis fails.
    """
    try:
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
        )

        audio_buffer = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_bytes = audio_buffer.getvalue()
    except Exception as e:
        logger.error("TTS synthesis failed: %s", e)
        raise RuntimeError(
            f"TTS synthesis failed (voice={voice}): {e}. "
            "This may be a transient Edge-TTS service issue. Retry shortly."
        ) from e

    if not audio_bytes:
        raise RuntimeError(
            f"TTS synthesis returned empty audio for voice={voice}. "
            "Check network connectivity to Edge-TTS service."
        )

    logger.info(
        "TTS synthesized %d bytes (voice=%s, text_len=%d)",
        len(audio_bytes), voice, len(text),
    )
    return audio_bytes


def synthesize_urdu(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    use_cache: bool = True,
) -> bytes:
    """Synchronous wrapper for Urdu speech synthesis.

    Generates an MP3 audio stream from Urdu text. Results are cached
    in-memory by text hash to prevent repeated synthesis of identical
    advisory strings (FR-5.3).

    Args:
        text: Urdu text to convert to speech.
        voice: Edge-TTS voice ID (default: ur-PK-AsadNeural).
        rate: Speaking rate adjustment.
        use_cache: Whether to use in-memory caching.

    Returns:
        MP3 audio bytes ready for playback.
    """
    if not text or not text.strip():
        logger.warning("TTS called with empty text — returning empty bytes")
        return b""

    # Cache key based on content + voice + rate
    cache_key = hashlib.md5(
        f"{text}|{voice}|{rate}".encode("utf-8")
    ).hexdigest()

    if use_cache and cache_key in _audio_cache:
        logger.debug("TTS cache hit for key=%s", cache_key[:8])
        return _audio_cache[cache_key]

    # Run async synthesis in a synchronous context
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an already-running event loop (e.g., Streamlit)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                audio_bytes = pool.submit(
                    asyncio.run,
                    _synthesize_async(text, voice, rate),
                ).result()
        else:
            audio_bytes = loop.run_until_complete(
                _synthesize_async(text, voice, rate)
            )
    except RuntimeError:
        # No event loop exists yet
        audio_bytes = asyncio.run(
            _synthesize_async(text, voice, rate)
        )

    # Cache the result
    if use_cache:
        _audio_cache[cache_key] = audio_bytes
        logger.debug("TTS cached result for key=%s (%d bytes)",
                     cache_key[:8], len(audio_bytes))

    return audio_bytes


def clear_cache() -> int:
    """Clear the in-memory TTS audio cache.

    Returns:
        Number of cache entries cleared.
    """
    global _audio_cache
    count = len(_audio_cache)
    _audio_cache = {}
    logger.info("TTS cache cleared (%d entries)", count)
    return count
