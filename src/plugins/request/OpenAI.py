from .utils import RequestABC, to_prompt_structure, to_instruction, to_json_desc, find_json
from openai import OpenAI as OpenAIClient
from Agently.utils import RuntimeCtxNamespace
import httpx
import time

class OpenAI(RequestABC):
    def __init__(self, request):
        self.request = request
        self.use_assistant = False
        self.assistant_id = None
        self.model_name = "OpenAI"
        self.model_settings = RuntimeCtxNamespace(f"model.{ self.model_name }", self.request.settings)
        self.request_type = self.request.request_runtime_ctx.get("request_type", "chat")
        if self.request_type == None:
            self.request_type = "chat"

    def _create_client(self):
        client_params = {}
        base_url = self.model_settings.get_trace_back("url")
        if base_url:
            client_params.update({ "base_url": base_url })
        proxy = self.request.settings.get_trace_back("proxy")
        if proxy:
            client_params.update({ "http_client": httpx.Client( proxies = proxy ) })
        api_key = self.model_settings.get_trace_back("auth.api_key")
        if api_key:
            client_params.update({ "api_key": api_key })
        else:
            raise Exception("[Request] OpenAI require api_key. use .set_auth({ 'api_key': '<Your-API-Key>' }) to set it.")
        client = OpenAIClient(**client_params)
        return client

    def construct_request_messages(self):
        #init request messages
        request_messages = []
        # - general instruction
        general_instruction_data = self.request.request_runtime_ctx.get_trace_back("prompt.general_instruction")
        if general_instruction_data:
            request_messages.append({ "role": "system", "content": f"[GENERAL INSTRUCTION]\n{ to_instruction(general_instruction_data) }" })
        # - role
        role_data = self.request.request_runtime_ctx.get_trace_back("prompt.role")
        if role_data:
            request_messages.append({ "role": "system", "content": f"[ROLE SETTINGS]\n{ to_instruction(role_data) }" })
        # - user info
        user_info_data = self.request.request_runtime_ctx.get_trace_back("prompt.user_info")
        if user_info_data:
            request_messages.append({ "role": "system", "content": f"[ABOUT USER]\n{ to_instruction(user_info_data) }" })
        # - headline
        headline_data = self.request.request_runtime_ctx.get_trace_back("prompt.headline")
        if headline_data:
            request_messages.append({ "role": "assistant", "content": to_instruction(headline_data) })
        # - chat history
        chat_history_data = self.request.request_runtime_ctx.get_trace_back("prompt.chat_history")
        if chat_history_data:
            request_messages.extend(chat_history_data)
        # - request message (prompt)
        prompt_input_data = self.request.request_runtime_ctx.get_trace_back("prompt.input")
        prompt_information_data = self.request.request_runtime_ctx.get_trace_back("prompt.information")
        prompt_instruction_data = self.request.request_runtime_ctx.get_trace_back("prompt.instruction")
        prompt_output_data = self.request.request_runtime_ctx.get_trace_back("prompt.output")
        # - files
        files_data = self.request.request_runtime_ctx.get_trace_back("prompt.files")
        # --- only input
        if not prompt_input_data and not prompt_information_data and not prompt_instruction_data and not prompt_output_data:
            raise Exception("[Request] Missing 'prompt.input', 'prompt.information', 'prompt.instruction', 'prompt.output' in request_runtime_ctx. At least set value to one of them.")
        prompt_text = ""
        if prompt_input_data and not prompt_information_data and not prompt_instruction_data and not prompt_output_data:
            prompt_text = to_instruction(prompt_input_data)
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
            prompt_text = to_prompt_structure(prompt_dict, end="[OUTPUT]:\n")
        # --- for vision model
        if self.request_type == "vision" and files_data:
            content = []
            content.append({ "type": "text", "text": prompt_text })
            for image_content in files_data:
                if isinstance(image_content, str):
                    content.append({ "type": "image_url", "image_url": { "url": image_content } })
                elif isinstance(image_content, dict):
                    content.append({ "type": "image_url", "image_url": image_content })
                else:
                    raise Exception(f"[Request] OpenAI vision model .files() can only accept string or dict.")
            request_messages.append({ "role": "user", "content": content })
        else:
            request_messages.append({ "role": "user", "content": prompt_text })
        return request_messages                

    def generate_request_data(self):
        self.use_assistant = self.model_settings.get_trace_back("use_assistant", False)
        if self.request_type == "chat" and self.use_assistant:
            assistant_id = self.model_settings.get_trace_back("assistant_id")
            if not assistant_id:
                raise Exception("[Request] OpenAI require 'assistant_id' when 'use_assistant' is True. Use agent.OpenAIAssistant.update() to create an assistant if you don't have one.")
            else:
                self.assistant_id = assistant_id
            options = self.model_settings.get_trace_back("options", {})
            if "model" not in options:
                options.update({ "model": "gpt-3.5-turbo" })
            return {
                "stream": True,
                "messages": self.construct_request_messages(),
                "options": options,
            }
        elif self.request_type == "chat":
            options = self.model_settings.get_trace_back("options", {})
            if "model" not in options:
                options.update({ "model": "gpt-3.5-turbo" })
            return {
                "stream": True,
                "messages": self.construct_request_messages(),
                **options
            }
        elif self.request_type == "vision":
            options = self.model_settings.get_trace_back("options", {})
            if "model" not in options:
                options.update({ "model": "gpt-4-vision-preview" })
            return {
                "stream": True,
                "messages": self.construct_request_messages(),
                **options
            }

    def request_assiatant(self, request_data: dict):
        client = self._create_client()
        # create thread
        thread = client.beta.threads.create(
            messages = request_data["messages"]
        )
        # create run
        run = client.beta.threads.runs.create(
            thread_id = thread.id,
            assistant_id = self.assistant_id,
            **request_data["options"]
        )
        # wait until run finished
        is_complete = False
        retrieved_run = None
        while not is_complete:
            time.sleep(1)
            retrieved_run = client.beta.threads.runs.retrieve(
                run_id = run.id,
                thread_id = thread.id,
            )
            if retrieved_run.status in ("completed", "requires_action"):
                is_complete = True
        # retrieve run step
        retrieved_run_steps = client.beta.threads.runs.steps.list(
            run_id = run.id,
            thread_id = thread.id,
        )
        message_id = None
        # find complete or requires_action step to get message_id
        for step in retrieved_run_steps.data:
            if step.status in ("completed", "requires_action"):
                message_id = step.step_details.message_creation.message_id
                break
        # retrieve message
        message = client.beta.threads.messages.retrieve(
            message_id = message_id,
            thread_id = thread.id,
        )
        # finally we get it...
        return message.content[0].text.value

    def request_gpt(self, request_data: dict):
        client = self._create_client()
        if self.request.request_runtime_ctx.get("response:type") == "JSON" and request_data["model"] in ("gpt-3.5-turbo-1106", "gpt-4-1106-preview"):
            request_data.update({ "response_format": { "type": "json_object" } })
        stream = client.chat.completions.create(
            **request_data
        )
        return stream

    def request_vision(self, request_data: dict):
        client = self._create_client()
        request_data["max_tokens"] = 4096
        stream = client.chat.completions.create(
            **request_data
        )
        return stream

    def request_model(self, request_data: dict):
        if self.use_assistant and self.request_type == "chat":
            return self.request_assiatant(request_data)
        elif self.request_type == "chat":
            return self.request_gpt(request_data)
        elif self.request_type == "vision":
            return self.request_vision(request_data)

    def broadcast_response_without_streaming(self, response):
        yield({ "event": "response:done", "data": response })

    def broadcast_response_with_streaming(self, response_generator):
        response_message = {}
        for part in response_generator:
            delta = dict(part.choices[0].delta)
            for key, value in delta.items():
                if key not in response_message:
                    response_message[key] = value or ""
                else:
                    response_message[key] += value or ""
            yield({ "event": "response:delta_origin", "data": part })
            yield({ "event": "response:delta", "data": part.choices[0].delta.content or "" })
        yield({ "event": "response:done_origin", "data": response_message })
        yield({ "event": "response:done", "data": response_message["content"] })

    def broadcast_response(self, response_generator):
        if self.use_assistant and self.request_type == "chat":
            return self.broadcast_response_without_streaming(response_generator)
        elif self.request_type == "chat":
            return self.broadcast_response_with_streaming(response_generator)
        elif self.request_type == "vision":
            return self.broadcast_response_with_streaming(response_generator)

    def export(self):
        return {
            "generate_request_data": self.generate_request_data,
            "request_model": self.request_model,
            "broadcast_response": self.broadcast_response,
        }

def export():
    return ("OpenAI", OpenAI)