import inspect
from .utils import ComponentABC

class Decorator(ComponentABC):
    def __init__(self, agent: object):
        self.agent = agent

    def auto_func(self, func: callable):
        def wrapper(*args, **kwargs):
            # generate input dict
            signature = inspect.signature(func)
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            input_dict = {}
            for param in signature.parameters:
                input_dict.update({ param: arguments.arguments[param] })
            # generate instruction
            instruction = inspect.getdoc(func)
            # generate output dict
            output_dict = signature.return_annotation
            return (
                self.agent
                    .input(input_dict)
                    .instruct(instruction)
                    .output(output_dict)
                    .start()
            )
        return wrapper

    def on_event(self, event: str):
        if not event.startswith("response:"):
            event = "response:" + event
        def decorator(func: callable):
            self.agent.add_event_listener(event, func)
        return decorator
        
    def export(self):
        return {
            "prefix": None,
            "suffix": None,
            "alias": {
                "auto_func": { "func": self.auto_func, "return_value": True },
                "on_event": { "func": self.on_event, "return_value": True },
            },
        }

def export():
    return ("Decorator", Decorator)