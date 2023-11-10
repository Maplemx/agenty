import os
from .utils import RequestABC, to_prompt_structure, to_instruction, to_json_desc, find_json
import erniebot

class Ernie(RequestABC):
    def __init__(self, request):
        self.request = request
        self.request_type = self.request.request_runtime_ctx.get("request_type", "chat")
        if self.request_type == None:
            self.request_type = "chat"

    def _create_client(self):
        if self.request_type == "chat":
            client = erniebot.ChatCompletion
        elif self.request_type == "embedding":
            client = erniebot.Embedding
        elif self.request_type == "image":
            client = erniebot.Image
        return client

    def construct_request_messages(self, request_runtime_ctx):
        #init request messages
        request_messages = []
        # - system message
        system_data = request_runtime_ctx.get("system")
        if system_data:
            request_messages.append({ "role": "user", "content": to_instruction(system_data) })
            request_messages.append({ "role": "assistant", "content": "好的，我明白了" })
        # - headline
        headline_data = request_runtime_ctx.get("headline")
        if headline_data:
            request_messages.append({ "role": "assistant", "content": to_instruction(headline_data) })
        # - chat history
        chat_history_data = request_runtime_ctx.get("chat_history")
        if chat_history_data:
            request_messages.extend(chat_history_data)
        # - request message (prompt)
        prompt_input_data = request_runtime_ctx.get("prompt_input")
        prompt_information_data = request_runtime_ctx.get("prompt_information")
        prompt_instruction_data = request_runtime_ctx.get("prompt_instruction")
        prompt_output_data = request_runtime_ctx.get("prompt_output")
        # --- only input
        if not prompt_input_data and not prompt_information_data and not prompt_instruction_data and not prompt_output_data:
            raise Exception("[Request] Missing 'prompt_input', 'prompt_information', 'prompt_instruction', 'prompt_output' in request runtime_ctx. At least set value to one of them.")
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
                    request_runtime_ctx.set("response:type", "JSON")
                else:
                    prompt_dict["[OUTPUT REQUIERMENT]"] = str(prompt_output_data)
            request_messages.append({ "role": "user", "content": to_prompt_structure(prompt_dict, end="[OUTPUT]:\n") })
        return request_messages

    def generate_request_data(self, get_settings, request_runtime_ctx):
        options = get_settings("model_settings.options", {})
        options = options if isinstance(options, dict) else {}
        access_token = get_settings("model_settings.auth", {})
        access_token = access_token if isinstance(access_token, dict) else {}
        request_data = {}
        # request type: chat
        if self.request_type == "chat":
            if "model" not in options:
                options["model"] = "ernie-bot-4"
            if "aistudio" not in access_token:
                raise Exception(f"[Request] ERNIE require 'access-token-for-aistudio' when request type is '{ self.request_type }'. Use .set_model_auth({{ 'aistudio': <YOUR-ACCESS-TOKEN-FOR-AISTUDIO> }}) to set.")
            erniebot.api_type = "aistudio"
            erniebot.access_token = access_token["aistudio"]
            request_data = {
                "messages": self.construct_request_messages(self.request.request_runtime_ctx),
                **options,
            }
        # request type: embedding
        elif self.request_type == "embedding":
            if "model" not in options:
                options["model"] = "ernie-text-embedding"
            if "aistudio" not in access_token:
                raise Exception(f"[Request] ERNIE require 'access-token-for-aistudio' when request type is '{ self.request_type }'. Use .set_model_auth({{ 'aistudio': <YOUR-ACCESS-TOKEN-FOR-AISTUDIO> }}) to set.\n How to get your own access token? visit: https://github.com/PaddlePaddle/ERNIE-Bot-SDK/blob/develop/docs/authentication.md")
            erniebot.api_type = "aistudio"
            erniebot.access_token = access_token["aistudio"]
            content_input = self.request.request_runtime_ctx.get("prompt_input")
            if not isinstance(content_input, list):
                content_input = [content_input]
            request_data = {
                "input": content_input,
                **options,
            }
        # request type: image
        elif self.request_type == "image":
            if "model" not in options:
                options["model"] = "ernie-vilg-v2"
            if "yinian" not in access_token:
                raise Exception(f"[Request] ERNIE require 'access-token-for-yinian' when request type is '{ self.request_type }'. Use .set_model_auth({{ 'yinian': <YOUR-ACCESS-TOKEN-FOR-YINIAN> }}) to set.\n⚠️ Yinian Access Token is different from AIStudio Access Token!\n How to get your own access token? visit: https://github.com/PaddlePaddle/ERNIE-Bot-SDK/blob/develop/docs/authentication.md")
            erniebot.api_type = "yinian"
            erniebot.access_token = access_token["yinian"]
            prompt = self.request.request_runtime_ctx.get("prompt_input")
            output_requirement = self.request_runtime_ctx.get("prompt_output", {})
            if not isinstance(output_requirement):
                output_requirement = {}
            if "width" not in output_requirement:
                output_requirement.update({ "width": 512 })
            if "height" not in output_requirement:
                output_requirement.update({ "height": 512 })
            request_data = {
                "prompt": prompt,
                **output_requirement,
                **options,
            }
        return request_data

    def request_model(self, request_data: dict):
        client = self._create_client()
        response = client.create(stream = True, **request_data)
        return response  

    def broadcast_response(self, response_generator):
        response_message = { "role": "assistant", "content": "" }
        full_response_message = {}
        for part in response_generator:
            full_response_message = dict(part)
            delta = part["result"]
            response_message["content"] += delta
            yield({ "event": "response:delta_origin", "data": part })
            yield({ "event": "response:delta", "data": delta })
        full_response_message["result"] = response_message["content"]
        yield({ "event": "response:done_origin", "data": full_response_message})
        yield({ "event": "response:done", "data": response_message["content"] })

    def export(self):
        return {
            "generate_request_data": self.generate_request_data,
            "request_model": self.request_model,
            "broadcast_response": self.broadcast_response,
        }

def export():
    return ("ERNIE", Ernie)