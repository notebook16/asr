"""
ASR gRPC server — VAD-based streaming. No sliding windows, no overlap.
Speech buffered until 500ms silence, then one transcribe per utterance.
"""
import sys
from pathlib import Path

_app_dir = Path(__file__).resolve().parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import grpc
from concurrent import futures
import asr_pb2
import asr_pb2_grpc

from session_manager import get_session
from whisper_engine import transcribe_to_text_with_timings
from streaming_asr import StreamingASR


class ASRService(asr_pb2_grpc.ASRServicer):

    def StreamAudio(self, request_iterator, context):
        client_id = dict(context.invocation_metadata()).get("client_id", "default")
        print(f"[ASR] StreamAudio started — client_id={client_id}")
        session = get_session(client_id)

        if not getattr(session, "streaming_asr", None):
            session.streaming_asr = StreamingASR(transcribe_to_text_with_timings)

        chunk_count = 0
        for request in request_iterator:
            chunk_count += 1
            chunk = request.audio_chunk
            if chunk:
                result = session.streaming_asr.process_audio_chunk(bytes(chunk))
                if result:
                    text = result["text"]
                    start_ts = result["startTs"]
                    end_ts = result["endTs"]
                    print(f"[ASR] 📤 utterance: {text[:60]}... startTs={start_ts} endTs={end_ts}")
                    yield asr_pb2.ASRResponse(
                        start_ts=float(start_ts),
                        end_ts=float(end_ts),
                        sentence=text,
                    )

        print(f"[ASR] StreamAudio ended — client_id={client_id} total_chunks={chunk_count}")


def serve():
    print("[ASR] Starting gRPC server on [::]:50051 ...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    asr_pb2_grpc.add_ASRServicer_to_server(ASRService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("[ASR] Server listening on port 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
