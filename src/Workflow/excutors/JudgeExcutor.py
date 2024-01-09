from ..lib.ChunkExecutorABC import ChunkExecutorABC
from ..lib.constants import EXCUTOR_TYPE_JUDGE

class JudgeExcutor(ChunkExecutorABC):
  def __init__(self, chunk_desc: dict, settings = {}):
    self.type = EXCUTOR_TYPE_JUDGE
    self.chunk = chunk_desc

  def exec(self, inputs_with_handle_name: dict, global_input: str):
    chunk_data = self.chunk['data']
    conditions = chunk_data['settings']['conditions']
    if len(inputs_with_handle_name) == 0:
      return {
        "status": "success",
        "dataset": {}
      }
    # 取第一个作为输入值
    for handle_name in inputs_with_handle_name:
      target_dep_val = inputs_with_handle_name[handle_name]
      break

    judge_res = {}
    has_positive_judge = False
    for condition in conditions:
      judge_val = False
      if condition['relation'] != 'others':
        compare_val_type = condition.get('value_type', 'string')
        # 解析匹配值
        compare_val = condition['value']
        if compare_val_type == 'list' or condition['relation'] in ['in', 'notIn']:
          compare_val = compare_val.split(',')
        elif compare_val_type == 'boolean':
          compare_val = compare_val == 'true'

        # 计算最后的匹配结果
        if condition['relation'] == '=':
          judge_val = target_dep_val == compare_val
        elif condition['relation'] == '!=':
          judge_val = target_dep_val != compare_val
        elif condition['relation'] == 'in':
          judge_val = target_dep_val in compare_val
        elif condition['relation'] == 'notIn':
          judge_val = target_dep_val not in compare_val
        elif condition['relation'] == 'startsWith':
          judge_val = target_dep_val.startswith(compare_val)
        elif condition['relation'] == 'endsWith':
          judge_val = target_dep_val.endswith(compare_val)
        elif condition['relation'] == 'contains':
          judge_val = target_dep_val.contains(compare_val)

      if judge_val:
        has_positive_judge = True
      judge_res[condition['id']] = judge_val
    # ”其它“ 仅在所有条件都未命中时才未真
    judge_res['others'] = not has_positive_judge
    return {
        "status": "success",
        "dataset": judge_res
    }
