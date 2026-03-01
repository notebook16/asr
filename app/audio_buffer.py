"""
Audio buffer for streaming/chunked audio input.
"""

BYTES_PER_SECOND = 32000  # 16kHz * 2 bytes (int16)
WINDOW_SIZE = int(0.5 * BYTES_PER_SECOND)   # 2.0 sec — stable for tiny model, reduces repetition
OVERLAP_SIZE = int(0.5 * BYTES_PER_SECOND)  # 0.4 sec overlap

def append_and_get_window(session, chunk):
    session.buffer.extend(chunk)
    total = len(session.buffer)

    if total >= WINDOW_SIZE:
        window = session.buffer[:WINDOW_SIZE]
        session.buffer = session.buffer[WINDOW_SIZE - OVERLAP_SIZE:]
        print(f"[Buffer] Window ready — was {total} bytes, kept overlap, remaining={len(session.buffer)} bytes")
        return window

    return None