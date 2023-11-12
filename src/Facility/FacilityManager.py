from ..utils import PluginManager, RuntimeCtx, RuntimeCtxNamespace
from .._global import global_plugin_manager, global_storage, global_settings

class FacilityManager(object):
    def __init__(self, *, storage: object=global_storage, parent_plugin_manager: object=global_plugin_manager, parent_settings: object = global_settings):
        # init plugin manager
        self.plugin_manager = PluginManager(parent = parent_plugin_manager)
        # use global storage
        self.storage = storage
        # init facility settings
        self.settings = RuntimeCtx(parent = global_settings)
        # install facilities
        self.refresh_plugins()

    def refresh_plugins(self):
        facilities = self.plugin_manager.get("facility")
        for facility_name, FacilityPluginClass in facilities.items():
            setattr(self, facility_name, FacilityPluginClass(storage = self.storage, plugin_manager = self.plugin_manager, settings = RuntimeCtxNamespace(facility_name, self.settings)))

    def set_settings(self, settings_key: str, settings_value: any):
        self.settings.set(settings_key, settings_value)
        return self

    def list(self):
        result = {}
        facility_names = [\
            attr_name for attr_name in dir(self)\
            if not attr_name.startswith("_")\
            and attr_name not in ("plugin_manager", "storage", "list")\
            and isinstance(getattr(self, attr_name), object)\
        ]
        for facility_name in facility_names:
            facility = getattr(self, facility_name)
            facility_method_names = [\
                method_name for method_name in dir(facility)\
                if not method_name.startswith("_")\
                and callable(getattr(facility, method_name))\
            ]
            result[facility_name] = facility_method_names
        return result