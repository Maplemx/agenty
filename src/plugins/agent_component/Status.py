from .utils import ComponentABC
from Agently.utils import RuntimeCtxNamespace

class Status(ComponentABC):
    def __init__(self, agent: object):
        self.agent = agent
        self.status_runtime_ctx = RuntimeCtxNamespace("status", self.agent.agent_runtime_ctx)
        self.status_mapping_runtime_ctx = RuntimeCtxNamespace("status_mapping", self.agent.agent_runtime_ctx)
        self.status_storage = self.agent.global_storage.table("status")
        self.settings = RuntimeCtxNamespace("component_settings.Status", self.agent.settings)
        self.global_status_namespace = None
        if self.settings.get_trace_back("auto"):
            self.load()

    def set(self, key: str, value: str):
        self.status_runtime_ctx.set(key, value)
        return self.agent

    def use_global_status(self, namespace_name: str=None):
        self.global_status_namespace = namespace_name if namespace_name else "default"
        return self.agent

    def append_mapping(self, status_key: str, status_value, alias_name: str, *args, **kwargs):
        self.status_mapping_runtime_ctx.append(
            f"{ status_key }.{ status_value }",
            {
                "alias_name": alias_name,
                "args": args,
                "kwargs": kwargs,
            }
        )
        return self.agent

    def save(self):
        status = self.status_runtime_ctx.get()
        self.status_storage.set(status).save()
        return self.agent

    def load(self):
        status = self.status_storage.get()
        self.status_runtime_ctx.update(status)
        return self.agent

    def _early(self):
        component_toggle = self.agent.settings.get_trace_back("component_toggles.Status")
        if component_toggle:
            # get mappings
            global_status_mappings_dict = self.agent.global_storage.table(f"status_mapping.{ self.global_status_namespace }").get() if self.global_status_namespace != None else {}
            agent_status_mappings_dict = self.status_mapping_runtime_ctx.get()
            if agent_status_mappings_dict == None:
                agent_status_mappings_dict = {}
            for status_key, status_value in self.status_runtime_ctx.get().items():
                # handle global mappings first
                if status_key in global_status_mappings_dict and status_value in global_status_mappings_dict[status_key]:
                    for global_status_mapping in global_status_mappings_dict[status_key][status_value]:
                        getattr(self.agent, global_status_mapping["alias_name"])(*global_status_mapping["args"], **global_status_mapping["kwargs"])
                # handle agent mappings
                if status_key in agent_status_mappings_dict and status_value in agent_status_mappings_dict[status_key]:
                    for agent_status_mapping in agent_status_mappings_dict[status_key][status_value]:
                        getattr(self.agent, agent_status_mapping["alias_name"])(*agent_status_mapping["args"], **agent_status_mapping["kwargs"])
            return None
        else:
            return None

    def export(self):
        return {
            "early": self._early,
            "prefix": None,
            "suffix": None,
            "alias": {
                "set_status": { "func": self.set },
                "save_status": { "func": self.save },
                "load_status": { "func": self.load },
                "use_global_status": { "func": self.use_global_status },
                "append_status_mapping": { "func": self.append_mapping },
            },
        }

def export():
    return ("Status", Status)