from .utils import ComponentABC

class ReplyReformer(ComponentABC):
    def __init__(self, agent: object):
        self.agent = agent
        self.reform_handler = None

    def add_reply_reform_handler(self, reform_handler: callable):
        self.reform_handler = reform_handler
        return self.agent

    def _suffix(self, event, data):
        if event == "response:finally" and callable(self.reform_handler):
            self.agent.request.response_cache["reply"] = self.reform_handler(data)

    def export(self):
        return {
            "prefix": None,
            "suffix": self._suffix,
            "alias": {
                "reform_reply": { "func": self.add_reply_reform_handler }
            },
        }

def export():
    return ("ReplyReformer", ReplyReformer)