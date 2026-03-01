"""
VAD-based streaming ASR. No sliding windows, no overlap.
Buffers audio while speech is detected; finalizes and transcribes after 500ms silence.
"""

import numpy as np

BYTES_PER_SECOND = 32000  # 16kHz, 16-bit mono
SILENCE_THRESHOLD_SEC = 0.5
RMS_SPEECH_THRESHOLD = 0.01


def is_speech(audio_chunk: bytes, threshold: float = RMS_SPEECH_THRESHOLD) -> bool:
    """Simple RMS-based voice activity. Returns True if chunk has enough energy."""
    if not audio_chunk or len(audio_chunk) < 2:
        return False
    arr = np.frombuffer(audio_chunk, dtype=np.int16)
    arr = arr.astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(np.square(arr)))
    return rms > threshold


class StreamingASR:
    """
    Voice-activity-driven streaming transcription.
    Buffers PCM while speech is detected; when silence >= 500ms, transcribes once and returns
    transcript with absolute utterance-level timestamps (startTs, endTs).
    """

    def __init__(self, transcribe_fn, silence_threshold_sec: float = SILENCE_THRESHOLD_SEC, rms_threshold: float = RMS_SPEECH_THRESHOLD):
        self.transcribe_fn = transcribe_fn
        self.silence_threshold_sec = silence_threshold_sec
        self.rms_threshold = rms_threshold
        self.speech_buffer = bytearray()
        self.silence_duration_sec = 0.0
        self.total_audio_seconds = 0.0  # global streaming clock (advances only during speech)

    def process_audio_chunk(self, chunk: bytes):
        """
        Process one PCM chunk (16kHz 16-bit mono).
        Returns None if still listening; returns {"text", "startTs", "endTs"} when utterance ends (500ms silence).
        """
        if not chunk:
            return None

        chunk_duration_sec = len(chunk) / BYTES_PER_SECOND

        if is_speech(chunk, self.rms_threshold):
            self.speech_buffer.extend(chunk)
            self.total_audio_seconds += chunk_duration_sec
            self.silence_duration_sec = 0.0
            return None

        # Silence
        if len(self.speech_buffer) == 0:
            return None

        self.silence_duration_sec += chunk_duration_sec
        if self.silence_duration_sec < self.silence_threshold_sec:
            return None

        # Finalize utterance: compute absolute timestamps
        audio = bytes(self.speech_buffer)
        utterance_duration_sec = len(audio) / BYTES_PER_SECOND
        utterance_start_base = self.total_audio_seconds - utterance_duration_sec

        self.speech_buffer.clear()
        self.silence_duration_sec = 0.0

        result = self.transcribe_fn(audio)
        if not result or not result.get("text"):
            return None

        start_rel = result.get("start", 0)
        end_rel = result.get("end", 0)
        absolute_start = utterance_start_base + start_rel
        absolute_end = utterance_start_base + end_rel

        # Ensure we never return zeros when we have speech
        if absolute_start <= 0 and absolute_end <= 0 and result.get("text"):
            absolute_start = max(0.0, utterance_start_base)
            absolute_end = absolute_start + max(0.0, end_rel - start_rel)

        # Debug
        print(f"[StreamingASR] segment.start={start_rel} segment.end={end_rel} total_audio_seconds={self.total_audio_seconds:.3f} utterance_duration={utterance_duration_sec:.3f}")
        print(f"[StreamingASR] utterance_start_base={utterance_start_base:.3f} absolute_start={absolute_start:.3f} absolute_end={absolute_end:.3f}")

        return {
            "text": result["text"].strip(),
            "startTs": round(absolute_start, 3),
            "endTs": round(absolute_end, 3),
        }
