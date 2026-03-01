"""
Session management for ASR requests (streaming sessions, cleanup).

Why:

->Each user needs isolated buffer

->We store previous text for dedup

"""



class Session:
    def __init__(self):
        self.buffer = bytearray()
        self.last_text = ""
        self.total_offset = 0.0
        self.last_emitted_words = []  # last 2 words, for repetition filter (max 2 same in a row)

sessions = {}

def get_session(client_id):
    if client_id not in sessions:
        sessions[client_id] = Session()
        print(f"[Session] New session created — client_id={client_id}")
    return sessions[client_id]