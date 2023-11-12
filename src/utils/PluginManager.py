import json
from .RuntimeCtx import RuntimeCtx
from .transform import find_json

class PluginManager(object):
    def __init__(self, *, parent: object=None):
        self.plugins_runtime_ctx = RuntimeCtx(parent = parent.plugins_runtime_ctx if parent else None)

    def register(self, module_name: str, plugin_name: str, plugin: callable):
        if module_name == "$":
            raise Exception("[Plugin Manager] Module name can not be '$'.")
        self.plugins_runtime_ctx.set(f"{ module_name }.{ plugin_name }", plugin)
        return self
    
    def set_settings(self, keys_with_dots: str, value: any):
        self.plugins_runtime_ctx.set(f"$.settings.{keys_with_dots  }", value)
        return self

    def update_settings(self, settings: dict):
        for key, value in settings.items():
            if value.lower() == "true":
                settings[key] = True
            if value.lower() == "false":
                settings[key] = False
            json_string = find_json(value)
            if json_string != None and json_string != '':
                settings[key] = json.loads(json_string)
            self.set_settings(key, settings[key])
        return self

    def get_settings(self, keys_with_dots: str=None, default: str=None):
        if keys_with_dots != None:
            return self.plugins_runtime_ctx.get_trace_back(f"$.settings.{ keys_with_dots }", default)
        else:
            return self.plugins_runtime_ctx.get_trace_back("$.settings", default)

    def get(self, module_name: str, plugin_name: str=None):
        plugins = self.plugins_runtime_ctx.get_trace_back()
        if module_name not in plugins:
            raise Exception(f"[Plugin Manager] Module '{ module_name }' is not in plugins runtime_ctx.")
        if plugin_name != None:
            if plugin_name not in plugins[module_name]:
                raise Exception(f"[Plugin Manager] Plugin '{ plugin_name }' is not in the plugins runtime_ctx of module '{ module_name }'.")
            return plugins[module_name][plugin_name]
        else:
            return plugins[module_name]

    def get_agent_component_list(self):
        agent_components = self.get("agent_component")
        return agent_components.keys()