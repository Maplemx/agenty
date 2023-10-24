import datetime
import json
import asyncio
from ..request import Request
from ..utils import RuntimeCtx, StorageDelegate, PluginManager, AliasManager, IdGenerator, to_json_desc, find_json, check_version

class Agent(object):
    def __init__(
        self,
        *,
        agent_id: str=None,
        parent_agent_runtime_ctx: object,
        global_storage: object,
        parent_plugin_manager: object
    ):
        # Integrate
        self.global_storage = global_storage
        self.plugin_manager = PluginManager(parent = parent_plugin_manager)
        self.alias_manager = AliasManager(self)
        self.agent_runtime_ctx = RuntimeCtx(parent = parent_agent_runtime_ctx)
        self.request_runtime_ctx = RuntimeCtx()
        self.request = Request(
            parent_plugin_manager = self.plugin_manager,
            parent_request_runtime_ctx = self.request_runtime_ctx,
        )
        # Agent Id
        if agent_id == None:
            self.agent_id = IdGenerator("agent").create()
        else:
            self.agent_id = agent_id
        # Agent Storage
        self.agent_storage = StorageDelegate(
            db_name = self.agent_id,
            plugin_manager = self.plugin_manager,
        )
        # Version Check In Debug Model
        if self.plugin_manager.get_settings("is_debug"):
            check_version_record = self.global_storage.get("agently", "check_version_record")
            today = str(datetime.date.today())
            if check_version_record != today:
                check_version()
                self.global_storage.set("agently", "check_version_record", today)
        # Agent Request Prefix & Suffix
        self.agent_request_prefix = []
        self.agent_request_suffix = []
        # Register Default Request Alias to Agent
        self.request._register_default_alias(self.alias_manager)
        # Agent Components
        agent_components = self.plugin_manager.get("agent_component")
        component_toggles = self.plugin_manager.get_settings("component_toggles")
        for agent_component_name, AgentComponentClass in agent_components.items():
            # Skip component_toggles.Those Be Toggled Off
            if agent_component_name in component_toggles and component_toggles[agent_component_name] == False:
                    continue
            # Attach Component
            agent_component_instance = AgentComponentClass(agent = self)
            setattr(self, agent_component_name, agent_component_instance)
            component_export = agent_component_instance.export()
            # Register export_prefix, export_suffix
            if component_export["prefix"]:
                if isinstance(component_export["prefix"], list):
                    self.agent_request_prefix.extend(component_export["prefix"])
                elif callable(component_export["prefix"]):
                    self.agent_request_prefix.append(component_export["prefix"])
            if component_export["suffix"]:
                if isinstance(component_export["suffix"], list):
                    self.agent_request_suffix.extend(component_export["suffix"])
                elif callable(component_export["suffix"]):
                    self.agent_request_suffix.append(component_export["suffix"])
            # Register Alias
            if component_export["alias"]:
                for alias_name, alias_info in component_export["alias"].items():
                    self.alias_manager.register(
                        alias_name,
                        alias_info["func"],
                        return_value = alias_info["return_value"] if "return_value" in alias_info else False,
                    )

    async def async_start(self):
        is_debug = self.plugin_manager.get_settings("is_debug")
        # Call Prefix Funcs to Prepare Prefix Data(From agent_runtime_ctx To request_runtime_ctx)
        for prefix_func in self.agent_request_prefix:
            prefix_data = await prefix_func() if asyncio.iscoroutinefunction(prefix_func) else prefix_func()
            if prefix_data != None:
                if isinstance(prefix_data, tuple) and isinstance(prefix_data[0], str) and prefix_data[1] != None:
                    self.request.request_runtime_ctx.update(prefix_data[0], prefix_data[1])
                else:
                    raise("[Agent Component] Prefix return data error: only accept None or Tuple('<Request Namespace>', <Update Dict>)")

        # Request
        event_generator = self.request.get_event_generator()
    
        # Call Suffix Func to Handle Response Events
        if is_debug:
            print("[Realtime Response]\n")
        async def call_request_suffix(response):
            for suffix_func in self.agent_request_suffix:
                if asyncio.iscoroutinefunction(suffix_func):
                    await suffix_func(response["event"], response["data"]) 
                else:
                    suffix_func(response["event"], response["data"])

        async for response in event_generator:
            if response["event"] == "response:delta" and is_debug:
                print(response["data"], end="")
            await call_request_suffix(response)
            if response["event"] == "response:done":
                if is_debug:
                    print("\n--------------------------\n")
                    print("[Final Response]\n", response["data"], "\n--------------------------\n")
                if self.request.response_cache["reply"] == None:
                    self.request.response_cache["reply"] = response["data"]

        await call_request_suffix({ "event": "response:finally", "data": self.request.response_cache })

        # Fix JSON if Required
        if self.request.response_cache["type"] == "JSON":
            try:
                self.request.response_cache["reply"] = json.loads(find_json(self.request.response_cache["reply"]))
            except json.JSONDecodeError as e:
                try:
                    fixed_result = await self.request\
                        .input({
                            "target": self.request.response_cache["prompt"]["input"],
                            "format": to_json_desc(self.request.response_cache["prompt"]["output"]),
                            "origin JSON String": self.request.response_cache["reply"] ,
                            "error": e.msg,
                            "position": e.pos,
                        })\
                        .output('Fixed JSON String can be parsed by Python only without explanation and decoration.')\
                        .get_result_async()
                    self.request.response_cache["reply"] = json.loads(find_json(fixed_result))
                except Exception as e:
                    raise(f"[Agent Request] Error still occured when try to fix JSON decode error: { str(e) }")

        self.request_runtime_ctx.empty()
        return self.request.response_cache["reply"]

    def start(self):
        reply = asyncio.run(self.async_start())
        return reply