from configparser import ConfigParser

from ..utils import PluginManager, RuntimeCtx, StorageDelegate
from ..global_plugin_manager import global_plugin_manager
from .Agent import Agent

class AgentFactory(object):
    def __init__(self, *, parent_plugin_manager:object = global_plugin_manager, is_debug = False):
        #runtime ctx
        self.agent_runtime_ctx = RuntimeCtx()

        #load global plugins
        self.plugin_manager = PluginManager(parent = parent_plugin_manager)

        #use storage delegate
        self.storage = StorageDelegate(
            db_name = "global",
            plugin_manager = self.plugin_manager,
        )

        #debug
        self.set_settings("is_debug", is_debug)

    def create_agent(self, agent_id = None):
        return Agent(
            agent_id = agent_id,
            parent_agent_runtime_ctx = self.agent_runtime_ctx,
            global_storage = self.storage,
            parent_plugin_manager = self.plugin_manager
        )

    def register_plugin(self, module_name: str, plugin_name: str, plugin: callable):
        self.plugin_manager.register(module_name, plugin_name, plugin)
        return self

    def set_settings(self, settings_key: str, settings_value: any):
        self.plugin_manager.set_settings(settings_key, settings_value)
        return self

    def toggle_component(self, component_name, is_enabled):
        self.plugin_manager.set_settings(f"component_toggles.{ component_name }", is_enabled)
        return self

    def set_proxy(self, proxy_setting: any):
        self.plugin_manager.set_settings("proxy", proxy_setting)
        return self