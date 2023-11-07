from .utils import ComponentABC

class Skill(ComponentABC):
    def __init__(self, agent):
        pass

    def export(self):
        return {
            "prefix": None,
            "suffix": None,
            "alias": {},
        }

def export():
    return ("Skill", Skill)