# 用于处理中断的逻辑
import time

BREAKING_TYPES = {
    "TIMEOUT": "timeout",
    "CALL_MAXIMUM": "call_maximum"
}


class BreakingHub:
    def __init__(self, breaking_handler, max_execution_limit=5, timeout_threshold=10):
        self.max_execution_limit = max_execution_limit
        self.timeout_threshold = timeout_threshold
        self.breaking_handler = breaking_handler
        self.execution_counts = {}

    # 节点执行次数记录器
    def recoder(self, chunk):
        chunk_id = chunk['id']
        if chunk_id not in self.execution_counts:
            self.execution_counts[chunk_id] = 1
        else:
            self.execution_counts[chunk_id] += 1

        if self.execution_counts[chunk_id] > self.max_execution_limit:
            self.handle_interrupt(chunk, BREAKING_TYPES["CALL_MAXIMUM"])

    def get_counts(self, chunk):
        chunk_id = chunk['id']
        return self.execution_counts.get(chunk_id, 0)

    def handle_interrupt(self, chunk, type):
        self.breaking_handler(chunk, type)
