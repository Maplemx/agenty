#ABC = Abstract Base Class
from abc import ABC, abstractmethod

class StorageABC(ABC):
    @abstractmethod
    def __init__(self, db_name: str):
        self.db_name = db_name

    #Methods for key-value storage
    @abstractmethod
    def set(self, table_name: str, key: str, value: any):
        pass

    @abstractmethod
    def set_all(self, table_name:str, full_data: dict):
        pass

    @abstractmethod
    def remove(self, table_name: str, key: str):
        pass

    @abstractmethod
    def update(self, table_name:str, update_data: dict):
        pass

    @abstractmethod
    def get(self, table_name: str, key: str):
        pass

    @abstractmethod
    def get_all(self, table_name: str, keys: (list, None)=None):
        pass