import uuid

def create_trace_id() -> str:
    """生成用于单次请求链路追踪的短 trace_id。"""
    return "req_" + str(uuid.uuid4().hex[:8])