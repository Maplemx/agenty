from abc import ABC, abstractmethod
from typing import Dict, Any

class ChunkExecutorABC(ABC):
    def __init__(self, chunk_desc: dict, settings = {}):
        self.chunk = chunk_desc
        self.settings = settings
        self.type = 'unknow'

    @abstractmethod
    def exec(self, inputs_with_handle_name: dict, global_input: str) -> Dict[str, Any]:
        """
        执行函数，要求返回 dict 结果，字段要求如下：
        1. status: 必须，表示执行状态，取值为 'success' 或 'error'
        2. dataset: 必须，执行结果，为 any 类型
        3. error_msg: 可选，错误信息，为 str 类型
        """
        pass
