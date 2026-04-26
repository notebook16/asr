#!/usr/bin/env bash
# Start ASR gRPC server on port 50051.
# Run from repo root: ./run.sh   or: cd asr-service && ./run.sh

cd "$(dirname "$0")"
if [ -d "venv" ]; then
  . venv/bin/activate
fi

# CPU-only streaming profile for t3.small (2 vCPU)
export ASR_MODEL_SIZE="${ASR_MODEL_SIZE:-base.en}"
export ASR_CPU_THREADS="${ASR_CPU_THREADS:-2}"
export ASR_BEAM_SIZE="${ASR_BEAM_SIZE:-2}"
export ASR_BEST_OF="${ASR_BEST_OF:-2}"
export ASR_WINDOW_SEC="${ASR_WINDOW_SEC:-2.0}"
export ASR_OVERLAP_SEC="${ASR_OVERLAP_SEC:-0.35}"
export ASR_SILENCE_SEC="${ASR_SILENCE_SEC:-0.6}"
export ASR_VAD_MODE="${ASR_VAD_MODE:-2}"
export ASR_VAD_MIN_RATIO="${ASR_VAD_MIN_RATIO:-0.35}"
export ASR_START_TRIGGER_CHUNKS="${ASR_START_TRIGGER_CHUNKS:-2}"
export ASR_PRE_ROLL_SEC="${ASR_PRE_ROLL_SEC:-0.3}"

python3 -m app.server
