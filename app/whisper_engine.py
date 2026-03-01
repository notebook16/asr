"""
Whisper-based speech recognition engine.
Model config fixed for VAD-based streaming (tiny, int8, cpu_threads=4).
"""

from faster_whisper import WhisperModel
import numpy as np

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8",
    cpu_threads=4,
)
print("[Whisper] Model loaded (tiny, int8, cpu_threads=4) — ready")


def transcribe(audio_bytes):
    """Returns list of segment dicts (for compatibility). Prefer transcribe_to_text for VAD streaming."""
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_np = audio_np.astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        audio_np,
        language="en",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=True,
        vad_filter=True,
        word_timestamps=True,
    )
    results = []
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        })
    return results


def transcribe_to_text(audio_bytes) -> str:
    """Transcribe audio to a single string."""
    out = transcribe_to_text_with_timings(audio_bytes)
    return out["text"] if out else ""


def transcribe_to_text_with_timings(audio_bytes):
    """
    Transcribe audio; return dict with text and relative start/end (seconds) for utterance.
    Used by StreamingASR for absolute timestamp computation.
    """
    if not audio_bytes or len(audio_bytes) < 1600:  # < 50ms
        return None
    print(f"[Whisper] Transcribing — {len(audio_bytes)} bytes ({len(audio_bytes) / 32000:.2f}s)")
    segments = transcribe(audio_bytes)
    start_rel = None
    end_rel = None
    text_parts = []
    for s in segments:
        if start_rel is None:
            start_rel = s.get("start", 0)
        end_rel = s.get("end", 0)
        text_parts.append(s.get("text", "").strip())
    text = " ".join(text_parts).strip()
    if not text:
        return None
    if start_rel is None:
        start_rel = 0.0
    if end_rel is None:
        end_rel = start_rel
    print(f"[Whisper] transcript: {text[:80]}{'...' if len(text) > 80 else ''} (rel time {start_rel:.2f}s–{end_rel:.2f}s)")
    return {"text": text, "start": start_rel, "end": end_rel}