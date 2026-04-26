"""
VAD-based streaming ASR with rolling windows.
Emits partial transcripts during speech using 2.0s windows with 0.35s overlap,
then flushes remaining audio after 500ms silence.
"""

import os
from vad_filter import AdvancedVAD

BYTES_PER_SECOND = 32000  # 16kHz, 16-bit mono
SILENCE_THRESHOLD_SEC = float(os.getenv("ASR_SILENCE_SEC", "0.5"))
WINDOW_SEC = float(os.getenv("ASR_WINDOW_SEC", "2.0"))
OVERLAP_SEC = float(os.getenv("ASR_OVERLAP_SEC", "0.35"))
WINDOW_BYTES = int(WINDOW_SEC * BYTES_PER_SECOND)
OVERLAP_BYTES = int(OVERLAP_SEC * BYTES_PER_SECOND)
STEP_SEC = WINDOW_SEC - OVERLAP_SEC
START_TRIGGER_CHUNKS = int(os.getenv("ASR_START_TRIGGER_CHUNKS", "2"))
PRE_ROLL_SEC = float(os.getenv("ASR_PRE_ROLL_SEC", "0.3"))
PRE_ROLL_BYTES = int(PRE_ROLL_SEC * BYTES_PER_SECOND)

if WINDOW_BYTES <= 0:
    raise ValueError("ASR_WINDOW_SEC must be > 0")
if OVERLAP_BYTES < 0 or OVERLAP_BYTES >= WINDOW_BYTES:
    raise ValueError("ASR_OVERLAP_SEC must be >= 0 and < ASR_WINDOW_SEC")
if START_TRIGGER_CHUNKS <= 0:
    raise ValueError("ASR_START_TRIGGER_CHUNKS must be > 0")


class StreamingASR:
    """
    Voice-activity-driven streaming transcription.
    Emits partial transcripts while speech is active, then flushes tail on silence.
    """

    def __init__(self, transcribe_fn, silence_threshold_sec: float = SILENCE_THRESHOLD_SEC):
        self.transcribe_fn = transcribe_fn
        self.silence_threshold_sec = silence_threshold_sec
        self.vad = AdvancedVAD()
        self.speech_buffer = bytearray()
        self.pre_roll_buffer = bytearray()
        self.silence_duration_sec = 0.0
        self.total_audio_seconds = 0.0  # global streaming clock (advances only during speech)
        self.window_start_in_utterance = 0.0
        self.last_emitted_text = ""
        self.speech_trigger_count = 0
        self.in_speech = False

    def _emit_window_result(self, window_audio: bytes, utterance_start_base: float):
        result = self.transcribe_fn(window_audio)
        if not result or not result.get("text"):
            return None

        start_rel = result.get("start", 0.0)
        end_rel = result.get("end", 0.0)
        absolute_start = utterance_start_base + self.window_start_in_utterance + start_rel
        absolute_end = utterance_start_base + self.window_start_in_utterance + end_rel
        text = result["text"].strip()

        if not text or text == self.last_emitted_text:
            return None

        self.last_emitted_text = text
        return {
            "text": text,
            "startTs": round(max(0.0, absolute_start), 3),
            "endTs": round(max(0.0, absolute_end), 3),
        }

    def _process_windows(self, emitted):
        utterance_duration_sec = len(self.speech_buffer) / BYTES_PER_SECOND
        utterance_start_base = self.total_audio_seconds - utterance_duration_sec

        while len(self.speech_buffer) >= WINDOW_BYTES:
            window_audio = bytes(self.speech_buffer[:WINDOW_BYTES])
            partial = self._emit_window_result(window_audio, utterance_start_base)
            if partial:
                emitted.append(partial)
            self.speech_buffer = self.speech_buffer[WINDOW_BYTES - OVERLAP_BYTES:]
            self.window_start_in_utterance += STEP_SEC

    def process_audio_chunk(self, chunk: bytes):
        """
        Process one PCM chunk (16kHz 16-bit mono).
        Returns a list of zero or more transcript segments:
        [{"text", "startTs", "endTs"}, ...]
        """
        emitted = []
        if not chunk:
            return emitted

        chunk_duration_sec = len(chunk) / BYTES_PER_SECOND
        speech_now = self.vad.is_speech(chunk)

        if speech_now:
            self.speech_trigger_count += 1
        else:
            self.speech_trigger_count = 0

        if not self.in_speech and self.speech_trigger_count >= START_TRIGGER_CHUNKS:
            self.in_speech = True
            if self.pre_roll_buffer:
                self.speech_buffer.extend(self.pre_roll_buffer)
                self.total_audio_seconds += len(self.pre_roll_buffer) / BYTES_PER_SECOND
                self.pre_roll_buffer.clear()

        if self.in_speech:
            self.speech_buffer.extend(chunk)
            self.total_audio_seconds += chunk_duration_sec
            if speech_now:
                self.silence_duration_sec = 0.0
            else:
                self.silence_duration_sec += chunk_duration_sec

            self._process_windows(emitted)
        else:
            self.pre_roll_buffer.extend(chunk)
            if len(self.pre_roll_buffer) > PRE_ROLL_BYTES:
                self.pre_roll_buffer = self.pre_roll_buffer[-PRE_ROLL_BYTES:]
            return emitted

        if not self.in_speech or self.silence_duration_sec < self.silence_threshold_sec:
            return emitted

        # Finalize utterance: transcribe any remaining tail audio once.
        audio = bytes(self.speech_buffer)
        remaining_duration_sec = len(audio) / BYTES_PER_SECOND
        utterance_start_base = self.total_audio_seconds - (self.window_start_in_utterance + remaining_duration_sec)

        self.speech_buffer.clear()
        self.silence_duration_sec = 0.0
        tail = self._emit_window_result(audio, utterance_start_base)
        if tail:
            emitted.append(tail)

        self.in_speech = False
        self.speech_trigger_count = 0
        self.window_start_in_utterance = 0.0
        self.last_emitted_text = ""
        return emitted
