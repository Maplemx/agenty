from .Request import Request
from .Agent import AgentFactory
from .Facility import FacilityManager
from ._global import global_plugin_manager, global_storage
from .utils import *

def create_agent(*args, **kwargs):
    return AgentFactory().create_agent(*args, **kwargs)

facility = FacilityManager()