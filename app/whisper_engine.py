"""
Whisper-based speech recognition engine.
Model config tuned for low-CPU streaming.
"""

from faster_whisper import WhisperModel
import numpy as np
import os

MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "base.en")
CPU_THREADS = int(os.getenv("ASR_CPU_THREADS", "2"))
BEAM_SIZE = int(os.getenv("ASR_BEAM_SIZE", "2"))
BEST_OF = int(os.getenv("ASR_BEST_OF", "2"))
LANGUAGE = os.getenv("ASR_LANGUAGE", "en")
model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8",
    cpu_threads=CPU_THREADS,
)
print(
    f"[Whisper] Model loaded ({MODEL_SIZE}, int8, cpu_threads={CPU_THREADS}, beam_size={BEAM_SIZE}, best_of={BEST_OF}) — ready"
)


def transcribe(audio_bytes):
    """Returns list of segment dicts (for compatibility). Prefer transcribe_to_text for VAD streaming."""
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_np = audio_np.astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        audio_np,
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        best_of=BEST_OF,
        temperature=0.0,
        # Disable long-context carry-over for chunked streaming stability.
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 350,
            "speech_pad_ms": 180,
        },
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