from .utils import RequestABC, to_prompt_structure, to_instruction, to_json_desc, find_json
from Agently.utils import RuntimeCtxNamespace
import httpx
import json

class Google(RequestABC):
    def __init__(self, request):
        self.request = request
        self.model_name = "Google"
        self.model_settings = RuntimeCtxNamespace(f"model.{ self.model_name }", self.request.settings)
        
    def construct_request_messages(self):
        #init request messages
        request_messages = []
        # - general instruction
        general_instruction_data = self.request.request_runtime_ctx.get_trace_back("prompt.general_instruction")
        if general_instruction_data:
            request_messages.append({ "role": "user", "parts": [{ "text": f"[YOUR GENERAL INSTRUCTION]?" }] })
            request_messages.append({ "role": "model", "parts": [{ "text": f"{ to_instruction(general_instruction_data) }" }] })
        # - role
        role_data = self.request.request_runtime_ctx.get_trace_back("prompt.role")
        if role_data:
            request_messages.append({ "role": "user", "parts": [{ "text": f"[YOUR ROLE SETTINGS]?" }] })
            request_messages.append({ "role": "model", "parts": [{ "text": f"{ to_instruction(role_data) }" }] })
        # - user info
        user_info_data = self.request.request_runtime_ctx.get_trace_back("prompt.user_info")
        if user_info_data:
            request_messages.append({ "role": "user", "parts": [{ "text": f"[USER INFO WHOM YOU'RE TALKING TO]?" }] })
            request_messages.append({ "role": "model", "parts": [{ "text": f"{ to_instruction(user_info_data) }" }] })
        # - headline
        headline_data = self.request.request_runtime_ctx.get_trace_back("prompt.headline")
        if headline_data:
            request_messages.append({ "role": "user", "parts": [{ "text": f"[HEADLINE ABOUT WHAT WE'RE TALKING ABOUT]?" }] })
            request_messages.append({ "role": "model", "parts": [{ "text": f"{ to_instruction(headline_data) }" }] })
        # - chat history
        chat_history_data = self.request.request_runtime_ctx.get_trace_back("prompt.chat_history")
        if chat_history_data:
            # make sure start with role "user"
            if chat_history_data[0]["role"] != "user":
                request_messages.append({ "role": "user", "parts": [{ "text": "[CHAT HISTORY]:" }] })
                current_role = "model"
            else:
                current_role = "user"
            for message in chat_history_data:
                if message["role"] != "user":
                    message["role"] = "model"
                if message["role"] == current_role:
                    request_messages.append({ "role": message["role"], "parts": [{ "text": message["content"] }] })
                else:
                    request_messages[-1]["parts"][0]["text"] += f"\n{ message['content'] }"
                current_role = "user" if current_role == "model" else "model"
            # make sure end with role "model"
            if request_messages[-1]["role"] != "model":
                request_messages.append({ "role": "model", "parts": [{ "text": "[CHAT HISTORY END]" }] })
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
                prompt_dict["[HELPFUL INFORMATION]"] = str(prompt_information_data)
            if prompt_instruction_data:
                prompt_dict["[INSTRUCTION]"] = to_instruction(prompt_instruction_data)
            if prompt_output_data:
                if isinstance(prompt_output_data, (dict, list, set)):
                    prompt_dict["[OUTPUT REQUIREMENT]"] = {
                        "TYPE": "JSON can be parsed in runtime",
                        "FORMAT": to_json_desc(prompt_output_data),
                    }
                    self.request.request_runtime_ctx.set("response:type", "JSON")
                else:
                    prompt_dict["[OUTPUT REQUIERMENT]"] = str(prompt_output_data)
            prompt_text = to_prompt_structure(prompt_dict, end="[OUTPUT]:\n")
        request_messages.append({ "role": "user", "parts": [{ "text": prompt_text }] })
        return request_messages

    def generate_request_data(self):
        options = self.model_settings.get_trace_back("options", {})
        return {
            "messages": self.construct_request_messages(),
            "options": options,
        }

    async def request_model(self, request_data: dict):
        api_key = self.model_settings.get_trace_back("auth.api_key")
        proxy = self.request.settings.get_trace_back("proxy")
        messages = request_data["messages"]
        options = request_data["options"]
        request_params = { 
            "data": json.dumps({ "contents": messages, **options }),
            "timeout": None,
        }
        client_params = {}
        if proxy:
            client_params["proxy"] = proxy
        async with httpx.AsyncClient(**client_params) as client:
            async with client.stream(
                "POST",
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:streamGenerateContent?key={ api_key }",
                **request_params
            ) as response:
                async for chunk in response.aiter_lines():
                    yield chunk

    async def broadcast_response(self, response_generator):
        full_content = ""
        buffer = ""
        async for chunk in response_generator:
            full_content += chunk
            if '"text": "' in chunk:
                content = json.loads(f"{{ {chunk} }}")
                yield({ "event": "response:delta", "data": content["text"] })
                buffer += content["text"]
        if buffer == "":
            raise Exception(f"[Request]Google Gemini Error: { full_content }")
        yield({ "event": "response:done_origin", "data": json.loads(full_content) })
        yield({ "event": "response:done", "data": buffer })

    def export(self):
        return {
            "generate_request_data": self.generate_request_data,
            "request_model": self.request_model,
            "broadcast_response": self.broadcast_response,
        }

def export():
    return ("Google", Google)