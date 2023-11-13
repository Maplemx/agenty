from .utils import ComponentABC
from Agently.utils import RuntimeCtxNamespace

class Role(ComponentABC):
    def __init__(self, agent: object):
        self.agent = agent
        self.user_info_runtime_ctx = RuntimeCtxNamespace("user_info", self.agent.agent_runtime_ctx)
        self.role_runtime_ctx = RuntimeCtxNamespace("role", self.agent.agent_runtime_ctx)
        self.role_storage = self.agent.global_storage.table("role")

    def __get_runtime_ctx(self, target: str):
        if target == "role":
            return self.role_runtime_ctx
        elif target == "user_info":
            return self.user_info_runtime_ctx

    def set_name(self, name: str, *, target: str):
        runtime_ctx = self.__get_runtime_ctx(target)
        runtime_ctx.set("NAME", name)
        return self.agent

    def set(self, key: any, value: any=None, *, target: str):
        runtime_ctx = self.__get_runtime_ctx(target)
        if value is not None:
            runtime_ctx.set(key, value)
        else:
            runtime_ctx.set("DESC", key)
        return self.agent

    def update(self, key: any, value: any=None, *, target: str):
        runtime_ctx = self.__get_runtime_ctx(target)
        if value is not None:
            runtime_ctx.update(key, value)
        else:
            runtime_ctx.update("DESC", key)
        return self.agent        

    def append(self, key: any, value: any=None, *, target: str):
        runtime_ctx = self.__get_runtime_ctx(target)
        if value is not None:
            runtime_ctx.append(key, value)
        else:
            runtime_ctx.append("DESC", key)
        return self.agent

    def extend(self, key: any, value: any=None, *, target: str):
        runtime_ctx = self.__get_runtime_ctx(target)
        if value is not None:
            runtime_ctx.extend(key, value)
        else:
            runtime_ctx.extend("DESC", key)
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
        for key, value in role_data.items():
            self.role_runtime_ctx.update(key, value)
        return self.agent

    def _prefix(self):
        component_toggle = self.agent.settings.get_trace_back("component_toggles.Role")
        if component_toggle:
            return {
                "role": self.role_runtime_ctx.get(),
                "user_info": self.user_info_runtime_ctx.get(),
            }
        else:
            return None

    def export(self):
        return {
            "prefix": self._prefix,
            "suffix": None,
            "alias": {
                "set_role_name": { "func": lambda name: self.set_name(name, target="role") },
                "set_role": { "func": lambda key, value=None: self.set(key, value, target="role") },
                "update_role": { "func": lambda key, value=None: self.update(key, value, target="role") },
                "append_role": { "func": lambda key, value=None: self.append(key, value, target="role") },
                "extend_role": { "func": lambda key, value=None: self.extend(key, value, target="role") },
                "set_user_name": { "func": lambda name: self.set_name(name, target="user_info") },
                "set_user_info": { "func": lambda key, value=None: self.set(key, value, target="user_info") },
                "update_user_info": { "func": lambda key, value=None: self.update(key, value, target="user_info") },
                "append_user_info": { "func": lambda key, value=None: self.append(key, value, target="user_info") },
                "extend_user_info": { "func": lambda key, value=None: self.extend(key, value, target="user_info") },
                "save_role": { "func": self.save },
                "load_role": { "func": self.load },
            },
        }

def export():
    return ("Role", Role)