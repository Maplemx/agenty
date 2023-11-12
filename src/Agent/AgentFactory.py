from configparser import ConfigParser

from ..utils import PluginManager, RuntimeCtx
from .._global import global_plugin_manager, global_storage, global_settings, global_websocket_server
from .Agent import Agent

class AgentFactory(object):
    def __init__(self, *, parent_plugin_manager:object = global_plugin_manager, is_debug = False):
        #runtime ctx
        self.factory_agent_runtime_ctx = RuntimeCtx()
        self.settings = RuntimeCtx(parent = global_settings)

        #use plugin manager
        self.plugin_manager = PluginManager(parent = parent_plugin_manager)

        #use global storage
        self.global_storage = global_storage

        #use global websocket server
        self.global_websocket_server = global_websocket_server

        #debug
        self.set_settings("is_debug", is_debug)

    def create_agent(self, agent_id = None):
        return Agent(
            agent_id = agent_id,
            parent_agent_runtime_ctx = self.factory_agent_runtime_ctx,
            global_storage = self.global_storage,
            global_websocket_server = self.global_websocket_server,
            parent_plugin_manager = self.plugin_manager,
            parent_settings = self.settings,
        )

    def register_plugin(self, module_name: str, plugin_name: str, plugin: callable):
        self.plugin_manager.register(module_name, plugin_name, plugin)
        return self

    def set_settings(self, settings_key: str, settings_value: any):
        self.settings.set(settings_key, settings_value)
        return self

    def toggle_component(self, component_name, is_enabled):
        self.set_settings(f"component_toggles.{ component_name }", is_enabled)
        return self

    def set_proxy(self, proxy_setting: any):
        self.set_settings("proxy", proxy_setting)
        return self