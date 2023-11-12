#ABC = Abstract Base Class
from abc import ABC

class FacilityABC(ABC):
    def __init__(self, *, storage: object, plugin_manager: object, settings: object):
        self.storage = storage
        self.plugin_manager = plugin_manager
        self.settings = settings