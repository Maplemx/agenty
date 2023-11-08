from .utils import ComponentABC

class Status(ComponentABC):
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