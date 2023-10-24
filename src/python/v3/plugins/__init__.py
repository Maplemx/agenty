import os
import sys
import importlib

def install_plugins(plugin_manager):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    for item in os.listdir(dir_path):
        if os.path.exists(f"{ dir_path }/{ item }/__init__.py"):
            module_plugins = importlib.import_module(f".{ item }", package = __package__)
            if not hasattr(module_plugins, "export"):
                raise Exception(f"[Plugin Manager] Function 'export' must be stated in '__init__.py' file in module plugin dir: { dir_path }/{ item }")
            module_plugin_export = getattr(module_plugins, "export")()
            module_plugin_list = module_plugin_export[0]
            module_default_settings = module_plugin_export[1]
            try:
                for plugin_info in module_plugin_list:
                    plugin_manager.register(plugin_info[0], plugin_info[1], plugin_info[2])
            except:
                raise Exception(f"[Plugin Manager] Function 'export' in '__init__.py' must return a list of tuple (<module name>, <plugin name>, <plugin method>).")
            if module_default_settings:
                plugin_manager.update_settings(module_default_settings)