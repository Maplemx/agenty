#ABC = Abstract Base Class
from Agently.utils import to_instruction, to_json_desc
from .transform import to_prompt_structure
from abc import ABC, abstractmethod

class RequestABC(ABC):
    def __init__(self, request):
        self.request = request
        self.model_name = "<Model-Name>"
        # if you want to use RuntimeCtxNamespace to help you manage your settings
        # from Agently.utils import RuntimeCtxNamespace
        # self.model_settings = RuntimeCtxNamespace(f"model.{ self.model_name }", self.request.settings)

    def construct_request_messages(self):
        '''
        This is an example to help you understand how to construct your own request message or prompt
        Here're Standard LLM Request Inputs from Agently that you can use :
        self.request.request_runtime_ctx.get("prompt.system")
        self.request.request_runtime_ctx.get("prompt.headline")
        self.request.request_runtime_ctx.get("prompt.chat_history")
        self.request.request_runtime_ctx.get("prompt.input")
        self.request.request_runtime_ctx.get("prompt.information")
        self.request.request_runtime_ctx.get("prompt.instruction")
        self.request.request_runtime_ctx.get("prompt.output")
        You can use them to construct request or prompt following the model rules.
        And down below is just one example that usually fits OpenAI messages rules.
        '''
        #init request messages
        request_messages = []
        # - system message
        system_data = self.request.request_runtime_ctx.get("prompt.system")
        if system_data:
            request_messages.append({ "role": "system", "content": to_instruction(system_data) })
        # - headline
        headline_data = self.request.request_runtime_ctx.get("prompt.headline")
        if headline_data:
            request_messages.append({ "role": "assistant", "content": to_instruction(headline_data) })
        # - chat history
        chat_history_data = self.request.request_runtime_ctx.get("prompt.chat_history")
        if chat_history_data:
            request_messages.extend(chat_history_data)
        # - request message (prompt)
        prompt_input_data = self.request.request_runtime_ctx.get("prompt.input")
        prompt_information_data = self.request.request_runtime_ctx.get("prompt.information")
        prompt_instruction_data = self.request.request_runtime_ctx.get("prompt.instruction")
        prompt_output_data = self.request.request_runtime_ctx.get("prompt.output")
        # --- only input
        if not prompt_input_data and not prompt_information_data and not prompt_instruction_data and not prompt_output_data:
            raise Exception("[Request] Missing 'prompt.input', 'prompt.information', 'prompt.instruction', 'prompt.output' in request_runtime_ctx. At least set value to one of them.")
        if prompt_input_data and not prompt_information_data and not prompt_instruction_data and not prompt_output_data:
            request_messages.append({ "role": "user", "content": to_instruction(prompt_input_data) })
        # --- construct prompt
        else:
            prompt_dict = {}
            if prompt_input_data:
                prompt_dict["[INPUT]"] = to_instruction(prompt_input_data)
            if prompt_information_data:
                prompt_dict["[HELPFUL INFORMATION]"] = to_instruction(prompt_information_data)
            if prompt_instruction_data:
                prompt_dict["[INSTRUCTION]"] = to_instruction(prompt_instruction_data)
            if prompt_output_data:
                if isinstance(prompt_output_data, (dict, list, set)):
                    prompt_dict["[OUTPUT REQUIREMENT]"] = {
                        "TYPE": "JSON can be parsed in Python",
                        "FORMAT": to_json_desc(prompt_output_data),
                    }
                    self.request.request_runtime_ctx.set("response:type", "JSON")
                else:
                    prompt_dict["[OUTPUT REQUIERMENT]"] = str(prompt_output_data)
            request_messages.append({ "role": "user", "content": to_prompt_structure(prompt_dict, end="[OUTPUT]:\n") })
        return request_messages

    def generate_request_data(self):
        # ...
        return {
            #Your request data dict
            #"stream": True,
            #"messages": [...],
        }

    def request_model(self, request_data: dict):
        return
            #response generator / response result

    def broadcast_response(self, response_generator):
        '''
        Use response_generator or response result to broadcast event data using yield
        Standard Event Data Format:
        * { "event": "response:delta", "data": any } (optional, important for streaming output)
        * { "event": "response:done", "data": any } (required)
        - { "event": "response:delta_origin", "data": any } (optional)
        - { "event": "response:done_origin", "data": any } (optional)
        If your response is not streaming, you can only yield response:done.
        '''
        return

    @abstractmethod
    def export(self):
        return {
            "generate_request_data": callable,# (get_settings, request_runtime_ctx) -> request_data: dict
            "request_model": callable,# (request_data) -> response_generator
            "broadcast_response": callable,# (response_generator) -> broadcast_event_generator
        }