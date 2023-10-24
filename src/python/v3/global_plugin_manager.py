from .utils import PluginManager
from .plugins import install_plugins

global_plugin_manager = PluginManager()
install_plugins(global_plugin_manager)