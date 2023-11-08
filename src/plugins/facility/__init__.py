import os
import importlib
import configparser

def export():
    plugin_list = []

    dir_path = os.path.dirname(os.path.abspath(__file__))
    dir_name = os.path.basename(dir_path)
    
    # read config.ini
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(f"{ dir_path }/config.ini")
    config_is_not_empty = len(dict(config)) > 1
    
    # load configs
    module_name = config["plugins"]["module_name"] if config_is_not_empty and "module_name" in config else dir_name
    exception_files = set(config["plugins"]["exception_files"]) if config_is_not_empty and "exception_files" in config["plugins"] else set([])
    exception_files.add("__init__.py")
    exception_dirs = set(config["plugins"]["exception_dirs"]) if config_is_not_empty and "exception_dirs" in config["plugins"] else set([])
    exception_dirs.add("utils")
    default_settings = dict(config["default settings"]) if config_is_not_empty and "default settings" in config else None

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