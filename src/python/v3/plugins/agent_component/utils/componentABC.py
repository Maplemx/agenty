#ABC = Abstract Base Class
from abc import ABC, abstractmethod

class componentABC(ABC):
    def __init__(self, agent):
        pass

    @abstractmethod
    def export(self):
        return {
            "prefix": callable or [callable],#()->(request_namespace: str, update_dict: dict)
            "suffix": callable,#(event: str, data: any)->None
            "alias": {
                alias_name: {
                    "func": callable,
                    "return_value": bool,# optional, default=False
                },
                # ...
            },
        }

