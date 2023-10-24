import os
import asyncio
import json
from configparser import ConfigParser

from ..utils import RuntimeCtx, RuntimeCtxNamespace, PluginManager, AliasManager, to_json_desc, find_json
from ..global_plugin_manager import global_plugin_manager

class Request(object):
    def __init__(
            self,
            *,
            parent_plugin_manager: object = global_plugin_manager,
            parent_request_runtime_ctx:object = None
        ):
        self.request_runtime_ctx = RuntimeCtx(parent = parent_request_runtime_ctx)
        self.response_cache = {
            "prompt": {},
            "type": None,
            "reply": None,
        }
        # Plugin Manager
        self.plugin_manager = PluginManager(parent = parent_plugin_manager)
        # Namespace
        self.model_settings = RuntimeCtxNamespace("$.settings.model_settings", self.plugin_manager.plugins_runtime_ctx, return_to = self)
        self.system = RuntimeCtxNamespace("system", self.request_runtime_ctx, return_to = self)
        self.headline = RuntimeCtxNamespace("headline", self.request_runtime_ctx, return_to = self)
        self.chat_history = RuntimeCtxNamespace("chat_history", self.request_runtime_ctx, return_to = self)
        self.prompt_input = RuntimeCtxNamespace("prompt_input", self.request_runtime_ctx, return_to = self)
        self.prompt_information = RuntimeCtxNamespace("prompt_information", self.request_runtime_ctx, return_to = self)
        self.prompt_instruction = RuntimeCtxNamespace("prompt_instruction", self.request_runtime_ctx, return_to = self)
        self.prompt_output = RuntimeCtxNamespace("prompt_output", self.request_runtime_ctx, return_to = self)
        # Alias
        self.alias_manager = AliasManager(self)
        self._register_default_alias(self.alias_manager)

    def _register_default_alias(self, alias_manager):
        alias_manager.register("set_model", lambda key, value: self.plugin_manager.set_settings(f"model_settings.{ key }", value))
        alias_manager.register("set_model_name", lambda model_name: self.plugin_manager.set_settings("model_settings.model_name", model_name))
        alias_manager.register("set_model_auth", lambda model_auth: self.plugin_manager.set_settings("model_settings.auth", model_auth))
        alias_manager.register("set_model_url", lambda model_url: self.plugin_manager.set_settings("model_settings.url", model_url))
        alias_manager.register("set_model_options", lambda model_options: self.plugin_manager.set_settings("model_settings.options", model_options))
        alias_manager.register("set_system", self.system.assign)
        alias_manager.register("set_headline", self.headline.assign)
        alias_manager.register("set_chat_history", self.chat_history.assign)
        alias_manager.register("input", self.prompt_input.assign)
        alias_manager.register("info", self.prompt_information.assign)
        alias_manager.register("instruct", self.prompt_instruction.assign)
        alias_manager.register("output", self.prompt_output.assign)
        alias_manager.register("set_proxy", lambda proxy_setting: self.plugin_manager.set_settings("proxy", proxy_setting))
        
    def get_event_generator(self):
        # Erase response cache
        self.response_cache = {
            "prompt": {},
            "type": None,
            "reply": None,
        }
        # Confirm model name
        model_name = self.plugin_manager.get_settings("model_settings.model_name")
        if not model_name:
            raise Exception(f"[Request] 'model_name' must be set into settings_runtime_ctx 'model_settings'.")
        model_name = model_name.lower()
        # Load request plugin by model name
        request_plugin = self.plugin_manager.get("request", model_name)
        # Generate request data
        request_data = request_plugin["generate_request_data"](
            request_runtime_ctx = self.request_runtime_ctx,
            get_settings = self.plugin_manager.get_settings,
        )
        if self.plugin_manager.get_settings("is_debug") == True:
            print("[Request Data]\n", json.dumps(request_data["data"], indent=4, ensure_ascii=False))
        # Cache simple prompt
        self.response_cache["prompt"]["input"] = self.prompt_input.get()
        self.response_cache["prompt"]["output"] = self.prompt_output.get()
        # Request and get response generator
        response_generator = request_plugin["request_model"](request_data)
        # Broadcast response
        broadcast_event_generator = request_plugin["broadcast_response"](response_generator)
        self.response_cache["type"] = self.request_runtime_ctx.get("response:type")
        # Reset request runtime_ctx
        self.request_runtime_ctx.empty()
        return broadcast_event_generator

    async def get_result_async(self):
        is_debug = self.plugin_manager.get_settings("is_debug")
        event_generator = self.get_event_generator()
        if is_debug:
            print("[Realtime Response]\n")
        async for response in event_generator:
            if response["event"] == "response:delta" and is_debug:
                print(response["data"], end="")
            if response["event"] == "response:done":
                if is_debug:
                    print("\n--------------------------\n")
                    print("[Final Response]\n", response["data"], "\n--------------------------\n")
                self.response_cache["reply"] = response["data"]
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
                    raise(f"[Agent Request] Error still occured when try to fix JSON decode error: { str(e) }")
        return self.response_cache["reply"]

    def get_result(self):
        reply = asyncio.run(self.get_result_async())
        return reply

    def start(self):
        return self.get_result()