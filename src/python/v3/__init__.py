from .agent import AgentFactory
from .request import Request
from .global_plugin_manager import global_plugin_manager
from .utils import *

def create_agent(*args, **kwargs):
    return AgentFactory().create_agent(*args, **kwargs)