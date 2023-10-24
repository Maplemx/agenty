import os
import importlib
import configparser

def export():
    plugin_list = []

    dir_path = os.path.dirname(os.path.abspath(__file__))
    
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(f"{ dir_path }/config.ini")
    default_settings = dict(config["default settings"]) if "default settings" in config else None
    module_name = config["plugins"]["module_name"]
    exception_files = config["plugins"]["exception_files"] if "exception_files" in config["plugins"] else []
    exception_dirs = config["plugins"]["exception_dirs"] if "exception_dirs" in config["plugins"] else []
    for item in os.listdir(dir_path):
        # import plugins in .py files
        if item.endswith('.py') and item not in exception_files:
            plugin = importlib.import_module(f".{ item[:-3] }", package = __package__)
            if not hasattr(plugin, "export"):
                raise Exception(f"[Plugin Manager] Function 'export' must be stated in plugin file: { dir_path }/{ item }")
            plugin_export = getattr(plugin, "export")()
            plugin_list.append((module_name, plugin_export[0], plugin_export[1]))
        # import plugins in dirs
        if os.path.exists(f"{ dir_path }/{ item }/__init__.py") and item not in exception_dirs:
            plugin = importlib.import_module(f".{ item }", package = __package__)
            if not hasattr(plugin, "export"):
                raise Exception(f"[Plugin Manager] Function 'export' must be stated in plugin dir's '__init__.py' file: { dir_path }/{ item }/__init__.py")
            plugin_export = getattr(plugin, "export")()
            plugin_list.append((module_name, plugin_export[0], plugin_export[1]))

    return (plugin_list, default_settings)