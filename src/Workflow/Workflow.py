from .MainExcutor import MainExcutor
from .utils.exec_tree import generate_exec_tree
from .Schema import Schema

class Workflow:
  def __init__(self, schema_data: dict, handler, settings = {}):
    """
    Workflow，初始参数 schema_data 形如 { 'chunks': [], 'edges': [] }，handler 为要处理响应的函数
    """
    self.handler = handler
    self.settings = settings
    self.schema = Schema(schema_data or {'chunks': [], 'edges': []})
    self.excutor = MainExcutor(handler, settings)
  
  def exec(self, input):
    exec_logic_tree = generate_exec_tree(self.schema)
    self.excutor.input(input, exec_logic_tree)
  
  def reset(self, schema_data: dict):
    self.schema = Schema(schema_data or {'chunks': [], 'edges': []})