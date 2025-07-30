import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../'))
import logging
import importlib
import sys
import os
import importlib.util



class Register:

    def __init__(self, registry_name):
        self._dict = {}
        self.dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value
        self.dict[key] = 1

    def __delitem__(self,key):
        if key in self._dict:
            del self._dict[key]
            del self.dict[key]


    def __call__(self, target):
        """Decorator to register a function or class."""
        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()


class Registers:
    
    block = Register('block')
    model = Register('model')

    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

def _handle_errors(errors):
    """Log out and possibly reraise errors during import."""
    if not errors:
        return
    for name, err in errors:
        logging.warning("Module {} import failed: {}".format(name, err))

def import_one_modules_for_register(module,package='blocks.code'):
    try:
        # module = package+'.'+module
        # importlib.import_module(module)
        if module in globals():
            print(f"{module} has")
            importlib.reload(globals()[module])
        else:
            print(f"{module} no")
            globals()[module] = importlib.import_module(module)
    except ImportError as error:
        return error
    return None

def import_all_modules_for_register(root_dir=None):
    """Import all modules for register."""
    if not root_dir:
        root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(root_dir)
    for key in ['base','stem','downsample']:
        code_dir = os.path.join(root_dir,'blocks',key,'code')
        os.makedirs(code_dir,exist_ok=True)
        if os.path.isdir(code_dir):
            sys.path.append(code_dir)
            for module in os.listdir(code_dir):
                if module.endswith('.py') and module!=os.path.basename(__file__):
                    module = module.strip('.py')
                    importlib.import_module(module)
    model_dir = os.path.join(root_dir,'models','code')
    os.makedirs(model_dir,exist_ok=True)
    if os.path.isdir(model_dir):
        sys.path.append(model_dir)
        for module in os.listdir(model_dir):
            if module.endswith('.py') and module!=os.path.basename(__file__):
                module = module[:-3]
                importlib.import_module(module)

def import_all_modules_for_register2(root_dir=None):
    """Import all modules for register."""
    if not root_dir:
        root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(root_dir)
    for file in os.listdir(root_dir):
        path = os.path.join(root_dir,file)
        if os.path.isdir(path):
            import_all_modules_for_register2(path)
        elif os.path.isfile(path) and file.endswith('.py') and file!=os.path.basename(__file__):
            module = file[:-3]
            try:
                importlib.import_module(module)
            except Exception as e:
                print(e)
                pass

def unload_modules_in_path(target_path):
    def unload_path(root_dir):
        if root_dir in sys.path:
            sys.path.remove(root_dir)
        for file in os.listdir(root_dir):
            path = os.path.join(root_dir,file)
            if os.path.isdir(path):
                unload_path(path)
    unload_path(target_path)
    modules_to_unload = [name for name, module in sys.modules.items() if hasattr(module, '__file__') and module.__file__ and target_path in module.__file__]
    for module_name in modules_to_unload:
        del sys.modules[module_name]
        del Registers.model[module_name]
        del Registers.block[module_name]

def import_module_from_path(module_name, file_path):
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

import_all_modules_for_register2()


