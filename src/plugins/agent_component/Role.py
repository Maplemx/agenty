from .utils import ComponentABC
from Agently.utils import RuntimeCtxNamespace

class Role(ComponentABC):
    def __init__(self, agent: object):
        self.agent = agent
        self.role_runtime_ctx = RuntimeCtxNamespace("role", self.agent.agent_runtime_ctx)
        self.role_storage = self.agent.global_storage.table("role")
        self.is_enabled = lambda: self.agent.settings.get_trace_back("component_toggles.Role")

    def toggle(self, is_enabled: bool):
        self.agent.settings.set("component_toggles.Role", is_enabled)
        self.agent.refresh_plugins()
        return self.agent

    def set_name(self, name: str):
        self.role_runtime_ctx.set("NAME", name)
        return self.agent

    def set(self, key: any, value: any=None):
        if value is not None:
            self.role_runtime_ctx.set(key, value)
        else:
            self.role_runtime_ctx.set("ROLE", key)
        return self.agent

    def update(self, key: any, value: any=None):
        if value is not None:
            self.role_runtime_ctx.update(key, value)
        else:
            self.role_runtime_ctx.update("ROLE", key)
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

    def save(self, role_name: str=None):
        if role_name == None:
            role_name = self.role_runtime_ctx.get("NAME")
        if role_name != None and role_name != "":
            self.role_storage\
                .set(role_name, self.role_runtime_ctx.get())\
                .save()
            return self.agent
        else:
            raise Exception("[Agent Component: Role] Role attr 'NAME' must be stated before save.")

    def load(self, role_name: str):
        role_data = self.role_storage.get(role_name)
        self.role_runtime_ctx.update(role_data)
        return self.agent

    def _prefix(self):
        if not self.is_enabled():
            return None
        role_data = self.role_runtime_ctx.get()
        if role_data != None:
            return {
                "system": role_data,
            }
        else:
            return None

    def export(self):
        return {
            "prefix": self._prefix,
            "suffix": None,
            "alias": {
                "toggle_role": { "func": self.toggle },
                "set_role_name": { "func": self.set_name },
                "set_role": { "func": self.set },
                "update_role": { "func": self.update },
                "append_role": { "func": self.append },
                "extend_role": { "func": self.extend },
                "save_role": { "func": self.save },
                "load_role": { "func": self.load },
            },
        }

def export():
    return ("Role", Role)