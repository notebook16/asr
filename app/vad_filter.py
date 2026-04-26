"""
Advanced VAD for 16kHz mono PCM streams.
Uses WebRTC VAD with lightweight energy checks and frame-ratio smoothing.
"""

import os
import numpy as np

try:
    import webrtcvad
except Exception:
    webrtcvad = None


class AdvancedVAD:
    def __init__(self):
        self.sample_rate = 16000
        self.frame_ms = int(os.getenv("ASR_VAD_FRAME_MS", "30"))
        self.mode = int(os.getenv("ASR_VAD_MODE", "2"))  # 0..3 (3 most aggressive)
        self.min_speech_ratio = float(os.getenv("ASR_VAD_MIN_RATIO", "0.35"))
        self.min_rms = float(os.getenv("ASR_VAD_MIN_RMS", "0.004"))

        if self.frame_ms not in (10, 20, 30):
            raise ValueError("ASR_VAD_FRAME_MS must be 10, 20, or 30")
        if self.mode < 0 or self.mode > 3:
            raise ValueError("ASR_VAD_MODE must be in range 0..3")

        self.frame_bytes = int(self.sample_rate * (self.frame_ms / 1000.0) * 2)
        self.vad = webrtcvad.Vad(self.mode) if webrtcvad else None
        if self.vad:
            print(
                f"[VAD] WebRTC VAD enabled (mode={self.mode}, frame_ms={self.frame_ms}, min_ratio={self.min_speech_ratio})"
            )
        else:
            print("[VAD] WebRTC VAD unavailable, falling back to RMS-only VAD")

    def _rms(self, audio_chunk: bytes) -> float:
        if not audio_chunk or len(audio_chunk) < 2:
            return 0.0
        arr = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(np.square(arr))))

    def is_speech(self, audio_chunk: bytes) -> bool:
        if not audio_chunk or len(audio_chunk) < self.frame_bytes:
            return False

        rms = self._rms(audio_chunk)
        if rms < self.min_rms:
            return False

        if not self.vad:
            return True

        total_frames = len(audio_chunk) // self.frame_bytes
        if total_frames == 0:
            return False

        speech_frames = 0
        for i in range(total_frames):
            frame = audio_chunk[i * self.frame_bytes : (i + 1) * self.frame_bytes]
            try:
                if self.vad.is_speech(frame, self.sample_rate):
                    speech_frames += 1
            except Exception:
                continue

        speech_ratio = speech_frames / total_frames
        return speech_ratio >= self.min_speech_ratio
