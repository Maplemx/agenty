# ABC = Abstract Base Class
from abc import ABC, abstractmethod


class RequestABC(ABC):
    @abstractmethod
    def __init__(self, request):
        self.request = request

    @abstractmethod
    def generate_request_data(self, get_settings, request_runtime_ctx):
        """
        使用该方法构造与大模型交互需要的数据。
        :param get_settings:
        :param request_runtime_ctx:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def request_model(self, request_data: dict):
        """
        使用该方法实现对大模型的请求交互。
        :param request_data:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def broadcast_response(self, response_generator):
        """
        使用该方法处理大模型的返回结果。
        :param response_generator:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def export(self):
        return {
            "generate_request_data": callable,
            # (get_settings, request_runtime_ctx) -> request_data: dict
            "request_model": callable,  # (request_data) -> response_generator
            "broadcast_response": callable,  # (response_generator) -> broadcast_event_generator
        }
