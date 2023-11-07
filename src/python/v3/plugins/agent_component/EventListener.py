from .utils import ComponentABC

class EventListener(ComponentABC):
    def __init__(self, agent: object):
        self.agent = agent
        self.listeners = {}

    def add(self, event:str, listener: callable):
        if event not in self.listeners:
            self.listeners[event] = []
        self.listeners[event].append(listener)
        return self.agent

    def on_delta(self, listener: callable):
        self.add("response:delta", listener)
        return self.agent

    def on_done(self, listener: callable):
        self.add("response:done", listener)
        return self.agent

    def on_finally(self, listener: callable):
        self.add("response:finally", listener)
        return self.agent

    def _call(self, event:str, data: any):
        if event in self.listeners:
            for listener in self.listeners[event]:
                listener(data)
        if event == "response:finally":
            self.listeners = {}

    def export(self):
        return {
            "prefix": None,
            "suffix": self._call,
            "alias": {
                "add_event_listener": { "func": self.add },
                "on_delta": { "func": self.on_delta },
                "on_done": { "func": self.on_done },
                "on_finally": { "func": self.on_finally },
            },
        }

def export():
    return ("EventListener", EventListener)