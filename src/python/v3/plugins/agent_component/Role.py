from .utils import componentABC
from Agently.utils import RuntimeCtxNamespace

class Role(componentABC):
    def __init__(self, agent: object):
        self.agent = agent
        self.role_runtime_ctx = RuntimeCtxNamespace("role", self.agent.agent_runtime_ctx)
        self.role_storage = self.agent.global_storage.table("role")
        self.is_enabled = lambda: self.agent.plugin_manager.get_settings("component_toggles.Role")

    def toggle(self, is_enabled: bool):
        self.agent.plugin_manager.set_settings("component_toggles.Role", is_enabled)
        return self.agent

    def set(self, key: any, value: any=None):
        if value is not None:
            self.role_runtime_ctx.set(key, value)
        else:
            self.role_runtime_ctx.set("ROLE", key)
        return self.agent

    def append(self, key: any, value: any=None):
        if value is not None:
            self.role_runtime_ctx.append(key, value)
        else:
            self.role_runtime_ctx.append("ROLE", key)
        return self.agent

    def extend(self, key: any, value: any=None):
        if value is not None:
            self.role_runtime_ctx.extend(key, value)
        else:
            self.role_runtime_ctx.extend("ROLE", key)
        return self.agent

    def save(self, role_name: str):
        self.role_storage\
            .set(role_name, self.role_runtime_ctx.get())\
            .save()
        return self.agent

    def load(self, role_name: str):
        role_data = self.role_storage.get(role_name)
        self.role_runtime_ctx.update(role_data)
        return self.agent

    def _prefix(self):
        if not self.is_enabled():
            return None
        role_data = self.role_runtime_ctx.get()
        if role_data != None:
            return ("system", role_data)
        else:
            return None

    def export(self):
        return {
            "prefix": self._prefix,
            "suffix": None,
            "alias": {
                "toggle_role": { "func": self.toggle },
                "set_role": { "func": self.set },
                "append_role": { "func": self.append },
                "extend_role": { "func": self.extend },
                "save_role": { "func": self.save },
                "load_role": { "func": self.load },
            },
        }

def export():
    return ("Role", Role)