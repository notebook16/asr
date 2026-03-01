#!/usr/bin/env bash
# Start ASR gRPC server on port 50051.
# Run from repo root: ./run.sh   or: cd asr-service && ./run.sh

cd "$(dirname "$0")"
if [ -d "venv" ]; then
  . venv/bin/activate
fi
python3 -m app.server
