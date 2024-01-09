from ..lib.ChunkExecutorABC import ChunkExecutorABC
from ..lib.constants import EXCUTOR_TYPE_END

class OutputExcutor(ChunkExecutorABC):
  def __init__(self, chunk_desc: dict, settings = {}):
    self.type = EXCUTOR_TYPE_END
    self.chunk = chunk_desc

  def exec(self, inputs_with_handle_name: dict, global_input: str):
    res = {}
    if len(inputs_with_handle_name) > 0:
      # 从各句柄读取数据，作为最终结果返回
      res = inputs_with_handle_name

    return {
        "status": "success",
        "dataset": res
    }
    
