import uuid

def create_trace_id() -> str:
    return "req_" + str(uuid.uuid4().hex[:8])