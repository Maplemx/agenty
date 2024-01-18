import json
from datetime import datetime
from .utils import ComponentABC
from Agently.utils import RuntimeCtxNamespace

class Tool(ComponentABC):
    def __init__(self, agent: object):
        self.agent = agent
        self.is_debug = self.agent.settings.get_trace_back("is_debug")
        self.settings = RuntimeCtxNamespace("plugin_settings.agent_component.Tool", self.agent.settings)
        self.tool_manager = self.agent.tool_manager
        self.tool_dict = {}
        self.must_call_tool_info = None
        self.call_tool_func = self.tool_manager.call_tool_func
        self.set_tool_proxy = self.tool_manager.set_tool_proxy

    def register_tool(self, tool_name: str, desc: any, args: dict, func: callable, *, categories: (str, list) = None, require_proxy: bool=False):
        self.tool_manager.register(tool_name, desc, args, func, categories=categories, require_proxy=require_proxy)
        self.tool_dict.update({ tool_name: { "tool_name": tool_name, "desc": desc, "args": args } })
        return self.agent

    def stop_tools(self, tool_name_list: (str, list)):
        if isinstance(tool_name_list, str):
            tool_name_list = [tool_name_list]
        for tool_name in tool_name_list:
            if tool_name in self.tool_dict:
                del self.tool_dict[tool_name]
        return self.agent

    def add_public_tools(self, tool_name_list: (str, list)):
        if isinstance(tool_name_list, str):
            tool_name_list = [tool_name_list]
        for tool_name in tool_name_list:
            tool_info = self.tool_manager.get_tool_info(tool_name, with_args=True)
            if tool_info:
                self.tool_dict.update({ tool_name: tool_info })
        return self.agent

    def add_public_categories(self, tool_category_list: (str, list)):
        if isinstance(tool_category_list, str):
            tool_category_list = [tool_category_list]
        tool_dict = self.tool_manager.get_tool_dict(categories=tool_category_list, with_args=True)
        for key, value in tool_dict.items():
            self.tool_dict.update({ key: value })
        return self.agent

    def add_all_public_tools(self):
        all_tool_dict = self.tool_manager.get_tool_dict(with_args=True)
        for key, value in all_tool_dict.items():
            self.tool_dict.update({ key: value })
        return self.agent

    def must_call(self, tool_name: str):
        tool_info = self.tool_manager.get_tool_info(tool_name, full=True)
        if tool_info:
            self.must_call_tool_info = tool_info
        else:
            raise Exception(f"[Agent Component] Tool-must call: can not find tool named '{ tool_name }'.")

    def call_plan_func(self, tool):
        if self.is_debug:
            print("[Agent Component] Using Tools: Start tool using judgement...")
        tool_list = []
        for tool_name, tool_info in self.tool_dict.items():
            tool_list.append(tool_info)
        result = (
            self.agent.worker_request
                .input({
                    "target": self.agent.request.request_runtime_ctx.get("prompt")
                })
                .info("current date", datetime.now().date())
                .info("tools", json.dumps(tool_list))
                .instruct("what tools to use for achieving {input.target}.\n * if use search tool, choose ONLY ONE SEARCH TOOL THAT FIT MOST and use the same language as {input.target}.")
                .output({
                    "tools_using": [{
                        "purpose": ("String", "what question you want to use tool to solve?"),
                        "using_tool": (
                            {
                                "tool_name": ("String", "{tool_name} from {tools}"),
                                "args": ("according {args} requirement in {tools}", ),
                            },
                            "output Null if do not need to use tool"
                        ),
                    }],
                })
                .start()
        )
        tool_results = {}
        for step in result["tools_using"]:
            if "using_tool" in step and isinstance(step["using_tool"], dict) and "tool_name" in step["using_tool"]:
                if self.is_debug:
                    print("[Using Tool]: ", step["using_tool"])
                tool_info = self.tool_manager.get_tool_info(step["using_tool"]["tool_name"], full=True)
                if tool_info:
                    tool_kwrags = step["using_tool"]["args"] if "args" in step["using_tool"] and isinstance(step["using_tool"]["args"], dict) else {}
                    if tool_info["require_proxy"]:
                        proxy = self.agent.settings.get_trace_back("proxy")
                        if proxy == None:
                            proxy = self.agent.tool_manager.get_tool_proxy()
                        if proxy:
                            tool_kwrags.update({ "proxy": proxy })
                    call_result = None
                    try:
                        call_result = self.call_tool_func(
                            tool_info["tool_name"],
                            **tool_kwrags
                        )
                    except Exception as e:
                        if self.is_debug:
                            print("[Tool Error]: ", e)
                    if call_result:
                        info_key = str(step["purpose"])
                        info_value = call_result["for_agent"] if isinstance(call_result, dict) and "for_agent" in call_result else call_result
                        tool_results[info_key] = info_value
                        if self.is_debug:
                            print("[Result]: ", info_key, info_value)
                else:
                    if self.is_debug:
                        print(f"[Result]: Can not find tool '{ step['using_tool']['tool_name'] }'")
        if len(tool_results.keys()) > 0:
            return tool_results
        else:
            return None

    def customize_call_plan(self, call_plan_func: callable):
        self.call_plan_func = call_plan_func
        return self.agent

    def _prefix(self):
        if self.must_call_tool_info:
            if self.is_debug:
                print(f"[Agent Component] Using Tools: Must call '{ self.must_call_tool_info['tool_name'] }'")
            self.agent.request.request_runtime_ctx.remove("prompt.instruct")
            self.agent.request.request_runtime_ctx.remove("prompt.output")
            return {
                "information": {
                    "function_must_call": {
                        "tool_name": self.must_call_tool_info["tool_name"],
                        "desc": self.must_call_tool_info["desc"],
                        "args": self.must_call_tool_info["args"],
                    },
                },
                "output": {
                    "can_call": ("Boolean", "all information above enough to generate all arguments in {function_must_call.args}?"),
                    "args": ("generate args dict according {function_must_call.args} requirement, leave argument value as null if you don't have enough information."),                    
                    "question": ("String", "if {can_call}==false, output question for user to collecting enough information."),
                }
            }
        elif len(self.tool_dict.keys()) > 0:
            tool_results = self.call_plan_func(self)
            if tool_results and len(tool_results.keys()) > 0:
                return {
                    "information": tool_results,
                    "instruction": ["using markdown format: [keywords](url) to mark info source in your answer."]
                }
            else:
                return None
        else:
            return None

    def export(self):
        return {
            "prefix": self._prefix,
            "suffix": None,
            "alias": {
                "register_tool": { "func": self.register_tool },
                "call_tool": { "func": self.call_tool_func },
                "set_tool_proxy": { "func": self.set_tool_proxy },
                "add_public_tools": { "func": self.add_public_tools },
                "use_public_tools": { "func": self.add_public_tools },
                "add_public_categories": { "func": self.add_public_categories },
                "use_public_categories": { "func": self.add_public_categories },
                "add_all_public_tools": { "func": self.add_all_public_tools },
                "use_all_public_tools": { "func": self.add_all_public_tools },
                "must_call": { "func": self.must_call },
                "customize_call_plan": { "func": self.customize_call_plan },
            }
        }

def export():
    return ("Tool", Tool)