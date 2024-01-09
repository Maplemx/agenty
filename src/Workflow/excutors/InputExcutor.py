from ..lib.ChunkExecutorABC import ChunkExecutorABC
from ..lib.constants import EXCUTOR_TYPE_START

class InputExcutor(ChunkExecutorABC):
  def __init__(self, chunk_desc: dict, settings = {}):
    self.type = EXCUTOR_TYPE_START
    self.chunk = chunk_desc

  def exec(self, inputs_with_handle_name: dict, global_input: str):
    return {
        "status": "success",
        "dataset": global_input
    }
