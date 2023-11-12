from .Request import Request
from .Agent import AgentFactory
from .Facility import FacilityManager
from .WebSocket import WebSocketServer, WebSocketClient
from ._global import global_plugin_manager, global_settings, global_storage, global_websocket_server
from .utils import *

def create_agent(*args, **kwargs):
    return AgentFactory().create_agent(*args, **kwargs)

facility = FacilityManager()