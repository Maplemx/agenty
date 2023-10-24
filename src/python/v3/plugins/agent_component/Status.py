from .utils import componentABC

class Status(componentABC):
    def __init__(self, agent):
        pass

    def export(self):
        return {
            "prefix": None,
            "suffix": None,
            "alias": {},
        }

def export():
    return ("Status", Status)