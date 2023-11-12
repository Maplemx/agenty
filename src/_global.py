from .utils import PluginManager, StorageDelegate, RuntimeCtx
from .plugins import install_plugins
from .WebSocket import WebSocketServer

global_plugin_manager = PluginManager()
global_settings = RuntimeCtx()
install_plugins(global_plugin_manager, global_settings)
global_storage = StorageDelegate(
    db_name = "global",
    plugin_manager = global_plugin_manager,
    settings = global_settings,
)
global_websocket_server = WebSocketServer()