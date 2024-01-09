from ..lib.ChunkExecutorABC import ChunkExecutorABC
from ..lib.constants import EXCUTOR_TYPE_END

class UnknowExcutor(ChunkExecutorABC):
  def __init__(self, chunk_desc: dict, settings = {}):
    self.type = EXCUTOR_TYPE_END
    self.chunk = chunk_desc
    self.settings = settings
    self.global_input_history = []

  def exec(self, inputs_with_handle_name: dict, global_input: str):
    chunk_data = self.chunk['data']
    return {
        "status": "error",
        "dataset": {},
        "error_msg": f"Unknow excutor '{chunk_data.get('title', '-')}'({chunk_data.get('type', '')})"
    }
    
