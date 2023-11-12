from .DataOps import DataOps

class StorageEditor(DataOps):
    def __init__(self, target_data: (dict, None), *, storage: object, table_name: str):
        self.storage = storage
        self.table_name = table_name
        super().__init__(target_data = target_data)

    def save(self):
        final_data = self.get()
        return self.storage.set_all(self.table_name, final_data)

class StorageDelegate(object):
    def __init__(self, *, db_name: str = "default", plugin_manager: object, settings: object):
        self.db_name = db_name
        self.plugin_manager = plugin_manager
        self.settings = settings
        return

    def set_storage_type(self, storage_type: str):
        self.settings.set("storage_type", storage_type)
        return self

    def __get_storage_plugin(self):
        storage_type = self.settings.get_trace_back("storage_type")
        return self.plugin_manager.get("storage", storage_type)(db_name = self.db_name)

    def set(self, table_name: str, key: str, value: any):
        self.__get_storage_plugin().set(table_name, key, value)
        return self

    def set_all(self, table_name: str, full_data: dict):
        self.__get_storage_plugin().set_all(table_name, full_data)
        return self

    def remove(self, table_name: str, key: str):
        self.__get_storage_plugin().remove(table_name, key)
        return self

    def update(self, table_name:str, update_data: dict):
        self.__get_storage_plugin().update(table_name, update_data)
        return self

    def get(self, table_name: str, key: str):
        return self.__get_storage_plugin().get(table_name, key)

    def get_all(self, table_name: str, keys: (list, None)=None):
        return self.__get_storage_plugin().get_all(table_name, keys)

    def table(self, table_name: str):
        table_data = self.get_all(table_name)
        return StorageEditor(table_data, storage = self, table_name = table_name)