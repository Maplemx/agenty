from .utils import FacilityABC
from Agently.utils import RuntimeCtx

class StatusManager(FacilityABC):
    def __init__(self, *, storage: object, plugin_manager: object, settings: object):
        self.storage = storage.table("status_mapping.default")
        self.plugin_manager = plugin_manager

    def set_status_namespace(self, namespace_name: str):
        self.storage = storage.table(f"status_mapping.{ namespace_name }")
        return self

    def append_mapping(self, status_key: str, status_value: str, alias_name: str, *args, **kwargs):
        self.storage\
            .append(
                f"{status_key}.{status_value}",
                {
                    "alias_name": alias_name,
                    "args": args,
                    "kwargs": kwargs,
                }
            )\
            .save()
        return self

    def get_mapping(self, status_key: str, status_value: str):
        return self.storage.get(f"{status_key}.{status_value}")

def export():
    return ("status_manager", StatusManager)