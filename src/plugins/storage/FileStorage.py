import os
import json
from .utils import StorageABC

class FileStorage(StorageABC):
    def __init__(self, db_name: str="default"):
        self.db_name = db_name
        if not os.path.exists("./file_storage"):
            os.mkdir("./file_storage")

    def __load_from_file(self, table_name: str):
        if not os.path.exists(f"./file_storage/{ self.db_name }/{ table_name }.data"):
            return {}
        else:
            with open(f"./file_storage/{ self.db_name }/{ table_name }.data", "r") as file:
                try:
                    result = json.loads(file.read())
                    return result
                except Exception as e:
                    print(f"[FileStorage] Error occurred when try to read file storage '{ self.db_name }/{ table_name }.data': { str(e) }")
                    print(f"[FileStorage] '{ self.db_name }/{ table_name }.data' will be reset to {{}}.")
                    self.__save_to_file(table_name, {})
                    return {}

    def __save_to_file(self, table_name: str, value: any):        
        if not os.path.exists(f"./file_storage/{ self.db_name }"):
            os.mkdir(f"./file_storage/{ self.db_name }")
        if not isinstance(value, str):
            value = json.dumps(value)
        with open(f"./file_storage/{ self.db_name }/{ table_name }.data", "w") as file:
            file.write(value)
            return True

    def set(self, table_name: str, key: str, value: any):
        table_data = self.__load_from_file(table_name)
        table_data.update({ key: json.dumps(value) })
        self.__save_to_file(table_name, table_data)

    def set_all(self, table_name:str, full_data: dict):
        for key, value in full_data.items():
            full_data[key] = json.dumps(value)
        self.__save_to_file(table_name, full_data)

    def remove(self, table_name: str, key: str):
        table_data = self.__load_from_file(table_name)
        if key in table_data:
            del table_data[key]
        self.__save_to_file(table_name, table_data)

    def update(self, table_name:str, update_data: dict):
        table_data = self.__load_from_file(table_name)
        for key, value in update_data.items():
            table_data.update({ key: json.dumps(value) })
        self.__save_to_file(table_name, table_data)

    def get(self, table_name: str, key: str):
        table_data = self.__load_from_file(table_name)
        if key in table_data:
            return json.loads(table_data[key])
        else:
            return None

    def get_all(self, table_name: str, keys: (list, None)=None):
        table_data = self.__load_from_file(table_name)
        if keys:            
            result = {}
            for key in keys:
                if key in table_data:
                    result.update({ key: json.loads(table_data[key]) })
                else:
                    result.update({ key: None })
            return result
        else:
            for key, value in table_data.items():
                table_data[key] = json.loads(value)
            return table_data

def export():
    return ("FileStorage", FileStorage)