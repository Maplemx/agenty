from .DataOps import DataOps, NamespaceOps

class RuntimeCtxNamespace(NamespaceOps):
    def __init__(self, namespace_name: str, runtime_ctx: object, *, return_to: object=None):
        super().__init__(namespace_name, runtime_ctx, return_to = return_to)

    def get_trace_back(self, keys_with_dots: (str, None) = None, default: str=None):
        return self.data_ops.get_trace_back(f"{ self.namespace_name }.{ keys_with_dots }" if keys_with_dots else self.namespace_name, default)

class RuntimeCtx(DataOps):
    def __init__ (self, *, parent: object=None, no_copy: bool=False):
        self.parent = parent
        self.runtime_ctx_storage = {}
        super().__init__(target_data = self.runtime_ctx_storage, no_copy = no_copy)

    def __update_trace_back_result(self, parent_result, result):
        for key in result:
            if key not in parent_result:
                parent_result[key] = {}
            elif not isinstance(parent_result[key], dict):
                parent_result[key] = {}
            if isinstance(result[key], dict):
                self.__update_trace_back_result(parent_result[key], result[key])
            else:
                parent_result[key] = result[key]
        return parent_result

    def get_trace_back(self, keys_with_dots: (str, None) = None, default: str=None):
        result = self.get(keys_with_dots)
        parent_result = self.parent.get_trace_back(keys_with_dots) if self.parent else None
        if result or parent_result:
            if isinstance(result, dict):
                parent_result = parent_result if isinstance(parent_result, dict) else {}
                return self.__update_trace_back_result(parent_result, result)
            else:
                return result if result else parent_result
        else:
            return default