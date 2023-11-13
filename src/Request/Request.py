import os
import asyncio
import threading
import queue
import json
import copy
from configparser import ConfigParser

from ..utils import RuntimeCtx, RuntimeCtxNamespace, PluginManager, AliasManager, to_json_desc, find_json
from .._global import global_plugin_manager, global_settings

class Request(object):
    def __init__(
            self,
            *,
            parent_plugin_manager: object = global_plugin_manager,
            parent_request_runtime_ctx: object = None,
            parent_settings: object = global_settings,
        ):
        self.request_runtime_ctx = RuntimeCtx(parent = parent_request_runtime_ctx)
        self.settings = RuntimeCtx(parent = parent_settings)
        self.response_cache = {
            "prompt": {},
            "type": None,
            "reply": None,
        }
        # Plugin Manager
        self.plugin_manager = PluginManager(parent = parent_plugin_manager)
        # Namespace
        self.model = RuntimeCtxNamespace("model", self.settings, return_to = self)
        self.prompt_general = RuntimeCtxNamespace("prompt.general", self.request_runtime_ctx, return_to = self)
        self.prompt_role = RuntimeCtxNamespace("prompt.role", self.request_runtime_ctx, return_to = self)
        self.prompt_user_info = RuntimeCtxNamespace("prompt.user_info", self.request_runtime_ctx, return_to = self)
        self.prompt_abstract = RuntimeCtxNamespace("prompt.abstract", self.request_runtime_ctx, return_to = self)
        self.prompt_chat_history = RuntimeCtxNamespace("prompt.chat_history", self.request_runtime_ctx, return_to = self)
        self.prompt_input = RuntimeCtxNamespace("prompt.input", self.request_runtime_ctx, return_to = self)
        self.prompt_information = RuntimeCtxNamespace("prompt.information", self.request_runtime_ctx, return_to = self)
        self.prompt_instruction = RuntimeCtxNamespace("prompt.instruction", self.request_runtime_ctx, return_to = self)
        self.prompt_output = RuntimeCtxNamespace("prompt.output", self.request_runtime_ctx, return_to = self)
        self.prompt_files = RuntimeCtxNamespace("prompt.files", self.request_runtime_ctx, return_to = self)
        # Alias
        self.alias_manager = AliasManager(self)
        self._register_default_alias(self.alias_manager)

    def set_settings(self, settings_key: str, settings_value: any):
        self.settings.set(settings_key, settings_value)
        return self

    def _register_default_alias(self, alias_manager):
        def set_model_settings(key: str, value: any, *, model_name:str = None):
            model_name = model_name if model_name else self.settings.get_trace_back("current_model")
            if model_name == None:
                raise Exception("[Model Settings] No model was appointed. Use .use_model(<model name>) or kwarg parameter model_name=<model_name> to set.")
            self.settings.update(f"model.{ model_name }.{ key }", value)

        alias_manager.register("use_model", lambda model_name: self.settings.set("current_model", model_name))
        alias_manager.register("set_model", set_model_settings)
        alias_manager.register("set_model_auth", lambda key, value, *, model_name=None: set_model_settings(f"auth.{ key }", value))
        alias_manager.register("set_model_url", lambda url, *, model_name=None: set_model_settings("url", url))
        alias_manager.register("set_model_option", lambda key, value, *, model_name=None: set_model_settings(f"options.{ key }", value))
        alias_manager.register("set_proxy", lambda proxy: self.settings.set("proxy", proxy))
        alias_manager.register("general", self.prompt_general.assign)
        alias_manager.register("role", self.prompt_role.assign)
        alias_manager.register("user_info", self.prompt_user_info.assign)
        alias_manager.register("abstract", self.prompt_abstract.assign)
        alias_manager.register("chat_history", self.prompt_chat_history.assign)
        alias_manager.register("input", self.prompt_input.assign)
        alias_manager.register("info", self.prompt_information.assign)
        alias_manager.register("instruct", self.prompt_instruction.assign)
        alias_manager.register("output", self.prompt_output.assign)
        alias_manager.register("files", self.prompt_files.assign)        
        
    async def get_event_generator(self, request_type: str=None):
        # Set Request Type
        self.request_runtime_ctx.set("request_type", request_type)
        # Erase response cache
        self.response_cache = {
            "prompt": {},
            "type": None,
            "reply": None,
        }
        # Confirm model name
        model_name = self.settings.get_trace_back("current_model")
        if not model_name:
            raise Exception(f"[Request] 'current_model' must be set. Use .use_model(<model_name>) to set.")
        # Load request plugin by model name
        request_plugin_instance = self.plugin_manager.get("request", model_name)(request = self)
        request_plugin_export = request_plugin_instance.export()
        # Generate request data
        if asyncio.iscoroutinefunction(request_plugin_export["generate_request_data"]):
            request_data = await request_plugin_export["generate_request_data"]()
        else:
            request_data = request_plugin_export["generate_request_data"]()
        if self.settings.get_trace_back("is_debug") == True:
            print("[Request Data]\n", json.dumps(request_data, indent=4, ensure_ascii=False))
        # Cache simple prompt
        self.response_cache["prompt"]["input"] = self.prompt_input.get()
        self.response_cache["prompt"]["output"] = self.prompt_output.get()
        # Request and get response generator
        if asyncio.iscoroutinefunction(request_plugin_export["request_model"]):
            response_generator = await request_plugin_export["request_model"](request_data)
        else:
            response_generator = request_plugin_export["request_model"](request_data)
        # Broadcast response
        if asyncio.iscoroutinefunction(request_plugin_export["broadcast_response"]):
            broadcast_event_generator = await request_plugin_export["broadcast_response"](response_generator)
        else:
            broadcast_event_generator = request_plugin_export["broadcast_response"](response_generator)        
        self.response_cache["type"] = self.request_runtime_ctx.get("response:type")
        # Reset request runtime_ctx
        self.request_runtime_ctx.empty()
        return broadcast_event_generator

    async def get_result_async(self, request_type: str=None):
        is_debug = self.settings.get_trace_back("is_debug")
        event_generator = await self.get_event_generator(request_type)
        if is_debug:
            print("[Realtime Response]\n")
        def handle_response(response):
            if response["event"] == "response:delta" and is_debug:
                print(response["data"], end="")
            if response["event"] == "response:done":
                if is_debug:
                    print("\n--------------------------\n")
                    print("[Final Response]\n", response["data"], "\n--------------------------\n")
                self.response_cache["reply"] = response["data"]

        if "__aiter__" in dir(event_generator):
            async for response in event_generator:
                handle_response(response)
        else:
            for response in event_generator:
                handle_response(response)
            
        if self.response_cache["type"] == "JSON":
            try:
                self.response_cache["reply"] = json.loads(find_json(self.response_cache["reply"]))
            except json.JSONDecodeError as e:
                try:
                    fixed_result = await self.start\
                        .input({
                            "target": self.response_cache["prompt"]["input"],
                            "format": to_json_desc(self.response_cache["prompt"]["output"]),
                            "origin JSON String": self.response_cache["reply"],
                            "error": e.msg,
                            "position": e.pos,
                        })\
                        .output('Fixed JSON String can be parsed by Python only without explanation and decoration.')\
                        .get_result_async()
                    fixed_result = json.loads(find_json(fixed_result))
                    return fixed_result
                except Exception as e:
                    raise Exception(f"[Agent Request] Error still occured when try to fix JSON decode error: { str(e) }")
        return self.response_cache["reply"]

    def get_result(self, request_type: str=None):
        reply_queue = queue.Queue()
        def start_in_theard():
            asyncio.set_event_loop(asyncio.new_event_loop())
            reply = asyncio.get_event_loop().run_until_complete(self.get_result_async(request_type))
            reply_queue.put_nowait(reply)
        theard = threading.Thread(target=start_in_theard)
        theard.start()
        theard.join()        
        reply = reply_queue.get_nowait()
        return reply

    def start(self):
        return self.get_result()