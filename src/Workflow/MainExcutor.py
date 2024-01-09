import json
from .BreakingHub import BreakingHub
from .excutors.UnknowExcutor import UnknowExcutor
from .utils.logger import get_default_logger
from .lib.constants import EXCUTOR_TYPE_JUDGE, EXCUTOR_TYPE_END
from .lib.ChunkExecutorABC import ChunkExecutorABC
from .excutors.AgentExcutor import AgentExcutor
from .excutors.InputExcutor import InputExcutor
from .excutors.OutputExcutor import OutputExcutor
from .excutors.JudgeExcutor import JudgeExcutor

class MainExcutor:
    def __init__(self, handler, settings={}):
        self.is_running = True
        self.running_status = 'idle'
        self.handler = handler
        self.settings = settings
        # 运行时过程中缓存的结果数据，结构为 { "type": "xxx", "value": "xxx" }
        self.runtime_res_store = {}
        # 执行次数记录，用于中断处理
        self.call_times_recorder = {}
        self.breaking_hub = BreakingHub(self._handle_breaking, 5)
        # 用户输入记录
        self.inputs = []
        self.current_global_input = ''
        self.UnknowExcutor = settings.get('UnknowExcutor', UnknowExcutor)
        self.logger = settings.get('logger', get_default_logger('Workflow'))
        # 已注册的执行器类型
        self.registed_excutors = {
            'Input': InputExcutor,
            'Output': OutputExcutor,
            'AgentRequest': AgentExcutor,
            'Judge': JudgeExcutor
        }
        # 判断类型的执行器
        self.judge_excutor_types = settings.get('judge_types', [EXCUTOR_TYPE_JUDGE])

    def input(self, input: str, chunks: list):
        self._reset_temp_status()
        self.current_global_input = input
        self.running_status = 'start'
        for chunk in chunks:
            self._chunks_walker(chunk)
        if self.running_status != 'finished':
            self._handle_not_finished()
        self.inputs.append(input)

    def regist_excutor(self, name: str, Excutor: ChunkExecutorABC):
        """
        注册执行器，传入执行器的名称及 Class（要求为 ChunkExecutorABC 的实现）
        """
        self.registed_excutors[name] = Excutor

    def unregist_excutor(self, name: str):
        """
        取消注册执行器，传入执行器的名称
        """
        if name in self.registed_excutors:
            del self.registed_excutors[name]

        return self

    def handle_command(self, data):
        """
        用于接收处理外部指令
        """
        if data is None or self.is_running == False or data["dataset"] is None:
            return

        command = data["dataset"]["command"]
        command_data = data["dataset"]["data"]
        if command == "input":
            if command_data and command_data["content"] and command_data['schema']:
                self.input(command_data["content"], command_data['schema'])
        elif command == "destroy":
            self.is_running = False

    def _get_chunk_excutor_class(self, name: str):
        """
        根据名称获取执行器的 Class
        """
        return self.registed_excutors.get(name, self.UnknowExcutor)

    def _reset_temp_status(self):
        """
        重置临时状态
        """
        self.running_status = 'idle'
        # 运行时过程中缓存的结果数据
        self.runtime_res_store = {}
        # 执行次数记录，用于中断处理
        self.call_times_recorder = {}
        self.breaking_hub = BreakingHub(self._handle_breaking, 5)
        self.current_global_input = ''

    def _get_chunk_res_setter_key(self, chunk):
        """
        获取存储数据的 key（需要考虑循环调用的场景）
        """
        return f"{chunk['id']}_{self.breaking_hub.get_counts(chunk) + 1}"

    def _get_chunk_res_getter_key(self, chunk):
        """
        获取读取数据的 key（需要考虑循环调用的场景）
        """
        return f"{chunk['id']}_{self.breaking_hub.get_counts(chunk)}"

    def _exec_chunk(self, chunk):
        """
        执行任务（执行到此处的都是上游数据已就绪了的）
        """
        # 提取出上游依赖的值(每个桩位仅取一个，如果遇到一个桩位有多个的值的情况，取最新执行的结果)
        deps_temp_res = {}
        for dep in chunk['deps']:
            dep_key = self._get_chunk_res_getter_key(dep)
            # 桩位名
            handler_name = dep['target_handler']
            # 处理数据
            if dep_key in self.runtime_res_store:
                dep_res = self.runtime_res_store[dep_key]
                new_res_count = self.breaking_hub.get_counts(dep)
                # 未取过值，或者有更新的值时，存储结果
                if handler_name not in deps_temp_res or new_res_count >= deps_temp_res[handler_name]['value_count']:
                    deps_temp_res[handler_name] = {
                        # 暂存执行结果
                        "full_value": dep_res['value'],
                        # 取对应的依赖的值
                        "value": dep_res['value'] if (dep['handler'] == 'output' or dep['handler'] is None) else dep_res['value'][dep['handler']],
                        # 暂存数据
                        "value_count": self.breaking_hub.get_counts(dep),
                        "handle_name": handler_name
                    }
        # 简化参数
        deps_dict = {}
        for handle_name in deps_temp_res:
            deps_dict[handle_name] = deps_temp_res[handle_name]['value']
        chunk_title = chunk["data"]["title"]

        # 交给执行器执行
        excutor_type = chunk['data']['type']
        ChunkExecutor = self._get_chunk_excutor_class(excutor_type)
        chunkExecutor = ChunkExecutor(chunk, self.settings)
        exec_res = chunkExecutor.exec(deps_dict, self.current_global_input)
        # 判断执行结果
        if exec_res['status'] == 'error':
            error_msg = exec_res.get('error_msg', 'Execution Error')
            # 主动中断执行
            raise Exception(
                f"Node Execution Exception: '{chunk_title}'({chunk['id']}) {error_msg}")
        # 存储数据
        self.runtime_res_store[self._get_chunk_res_setter_key(chunk)] = {
            'type': excutor_type,
            'value': exec_res.get('dataset')
        }
        # 判断是否执行结束
        if chunkExecutor.type == EXCUTOR_TYPE_END:
            self._handle_finished(exec_res.get('dataset'))

    def _check_dep_ready(self, chunk):
        """
        监测上游依赖是否就绪
        """
        deps = chunk.get('deps', [])
        if len(deps) == 0:
            return True
        dep_handlers = {}
        for dep in deps:
            # 记录桩点位
            if dep['target_handler'] not in dep_handlers:
                dep_handlers[dep['target_handler']] = False
            # 判断数据是否就绪
            if self._get_chunk_res_getter_key(dep) in self.runtime_res_store:
                dep_handlers[dep['target_handler']] = True
        # 判断数据是否就绪
        for dep_status in dep_handlers:
            if dep_handlers[dep_status] == False:
                return False
        return True

    def _check_branch_access(self, chunk):
        """
        判断当前chunk分支是否可执行（当父级为判断，如条件不符合则中断）
        """
        deps = chunk.get('deps', [])
        if len(deps) == 0:
            return True
        dep_handlers = {}
        for dep in deps:
            # 判断数据是否就绪
            key = self._get_chunk_res_getter_key(dep)
            # 如果数据未就绪，表示数据依赖使用的是另外的分组逻辑
            if key not in self.runtime_res_store:
                continue
            # 如果某个桩位的所有上游都为 judge 结果，则判定是否有其中一个分支是通路
            res = self.runtime_res_store[key]
            # 针对判断的场景，取判断结果来路由
            if res['type'] in self.judge_excutor_types:
                # 记录桩点位
                if dep['target_handler'] not in dep_handlers:
                    dep_handlers[dep['target_handler']] = False
                # 判断结果
                judge_res = res.get('value', {})
                # 如果判断结果为 True，则更新该桩位的状态为 True
                if judge_res.get(dep['handler'], False) == True:
                    dep_handlers[dep['target_handler']] = True
        # 判断所有桩位，看是否存在都为 False 的情况（此时分支中断）
        for dep_status in dep_handlers:
            if dep_handlers[dep_status] == False:
                return False
        return True

    def _chunks_walker(self, chunk):
        """
        执行整个 flow
        """
        # 校验上游依赖是否就绪
        if not self._check_dep_ready(chunk):
            self.logger.info(f"'{chunk['data']['title']}' dependencies ready.")
            return
        self.logger.info(f"'{chunk['data']['title']}' dependencies ready.")
        # 校验单桩位的情况，分支是否可执行（例如条件判断未满足，分支不执行）
        if not self._check_branch_access(chunk):
            self.logger.info(
                f"'{chunk['data']['title']}' conditions not met")
            return
        # 执行当前 chunk
        self._exec_chunk(chunk)
        # 记录执行
        self.breaking_hub.recoder(chunk)
        # 再执行分支 chunk
        for branch_chunk in chunk['branches']:
            self._chunks_walker(branch_chunk)

    def _handle_finished(self, res):
        self.running_status = 'finished'
        self.logger.info("[Finished]")
        self.handler("response", res)

    def _handle_not_finished(self):
        self.handler("error", 'Workflow not yet completed')

    def _handle_breaking(self, chunk):
        """
        处理中断
        """
        self.logger.error(f"Exceeded maximum execution limit: {chunk.id}")
        # 中断之前处理相关逻辑
        # 主动中断执行
        raise Exception(f"Exceeded maximum execution limit: {chunk.id}")
