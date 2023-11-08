import json
import inspect

class AliasManager(object):
    def __init__(self, target: object):
        self.target = target
        self.alias_name_list = []
        self.alias_details = {}

    def register(self, alias_name: str, alias_func: callable, *, return_value: bool=False):
        if alias_name in self.alias_name_list:
            raise Exception(f"[Alias Manager] alias_name '{ alias_name }' has already been registered.")
        if return_value:
            setattr(self.target, alias_name, alias_func)
        else:
            def _alias_func(*args, **kwargs):
                alias_func(*args, **kwargs)
                return self.target
            setattr(self.target, alias_name, _alias_func)
        self.alias_name_list.append(alias_name)
        self.alias_details.update({ alias_name: { "func": alias_func, "return_value": return_value } })

    def get_alias_info(self):
        result = {}
        for alias_name, alias_info in self.alias_details.items():
            result[alias_name] = { "paramaters": [] }
            alias_func_info = inspect.signature(alias_info["func"])
            for name, meta in alias_func_info.parameters.items():
                result[alias_name]["paramaters"].append(str(meta.name))
            result[alias_name]["is_return_value"] = alias_info["return_value"]
            result[alias_name]["docstring"] = str(alias_info["func"].__doc__)
        return result

    def print_alias_info(self):
        print(json.dumps(self.get_alias_info(), indent=4, ensure_ascii=False))