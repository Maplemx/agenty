#ABC = Abstract Base Class
from abc import ABC, abstractmethod

class RequestABC(ABC):
    def __init__(self, request):
        self.request = request

    @abstractmethod
    def export(self):
        return {
            "generate_request_data": callable,# (get_settings, request_runtime_ctx) -> request_data: dict
            "request_model": callable,# (request_data) -> response_generator
            "broadcast_response": callable,# (response_generator) -> broadcast_event_generator
        }