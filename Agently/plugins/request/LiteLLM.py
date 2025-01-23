from litellm import acompletion
from .utils import RequestABC, to_prompt_structure, to_instruction, to_json_desc, find_json, format_request_messages
from Agently.utils import RuntimeCtxNamespace

class LiteLLM(RequestABC):
    def __init__(self, request):
        self.request = request
        self.model_name = "LiteLLM"
        self.model_settings = RuntimeCtxNamespace(f"model.{self.model_name}", self.request.settings)
        self.request_type = self.request.request_runtime_ctx.get("request_type", "chat")
        self.default_options = {
            "stream": True
        }
        if not self.model_settings.get_trace_back("message_rules.no_multi_system_messages"):
            self.model_settings.set("message_rules.no_multi_system_messages", True)
        if not self.model_settings.get_trace_back("message_rules.strict_orders"):
            self.model_settings.set("message_rules.strict_orders", True)
        if not self.model_settings.get_trace_back("message_rules.no_multi_type_messages"):
            self.model_settings.set("message_rules.no_multi_type_messages", True)

    def construct_request_messages(self):
        #init request messages
        request_messages = []
        # - general instruction
        general_instruction_data = self.request.request_runtime_ctx.get_trace_back("prompt.general")
        if general_instruction_data:
            request_messages.append({"role": "system", "content": [{"type": "text", "text": f"[GENERAL INSTRUCTION]\n{to_instruction(general_instruction_data)}"}]})
        # - role
        role_data = self.request.request_runtime_ctx.get_trace_back("prompt.role")
        if role_data:
            request_messages.append({"role": "system", "content": [{"type": "text", "text": f"[ROLE SETTINGS]\n{to_instruction(role_data)}"}]})
        # - user info
        user_info_data = self.request.request_runtime_ctx.get_trace_back("prompt.user_info")
        if user_info_data:
            request_messages.append({"role": "system", "content": [{"type": "text", "text": f"[ABOUT USER]\n{to_instruction(user_info_data)}"}]})
        # - abstract
        headline_data = self.request.request_runtime_ctx.get_trace_back("prompt.abstract")
        if headline_data:
            request_messages.append({"role": "assistant", "content": [{"type": "text", "text": f"[ABSTRACT]\n{to_instruction(headline_data)}"}]})
        # - chat history
        chat_history_data = self.request.request_runtime_ctx.get_trace_back("prompt.chat_history")
        if chat_history_data:
            request_messages.extend(chat_history_data)
        # - request message (prompt)
        prompt_input_data = self.request.request_runtime_ctx.get_trace_back("prompt.input")
        prompt_info_data = self.request.request_runtime_ctx.get_trace_back("prompt.info")
        prompt_instruct_data = self.request.request_runtime_ctx.get_trace_back("prompt.instruct")
        prompt_output_data = self.request.request_runtime_ctx.get_trace_back("prompt.output")
        # --- only input
        if not prompt_input_data and not prompt_info_data and not prompt_instruct_data and not prompt_output_data:
            raise Exception("[Request] Missing 'prompt.input', 'prompt.info', 'prompt.instruct', 'prompt.output' in request_runtime_ctx. At least set value to one of them.")
        prompt_text = ""
        if prompt_input_data and not prompt_info_data and not prompt_instruct_data and not prompt_output_data:
            prompt_text = to_instruction(prompt_input_data)
        # --- construct prompt
        else:
            prompt_dict = {}
            if prompt_input_data:
                prompt_dict["[INPUT]"] = to_instruction(prompt_input_data)
            if prompt_info_data:
                prompt_dict["[HELPFUL INFORMATION]"] = str(prompt_info_data)
            if prompt_instruct_data:
                prompt_dict["[INSTRUCTION]"] = to_instruction(prompt_instruct_data)
            if prompt_output_data:
                if isinstance(prompt_output_data, (dict, list, set)):
                    prompt_dict["[OUTPUT REQUIREMENT]"] = {
                        "TYPE": "JSON String can be parsed",
                        "FORMAT": to_json_desc(prompt_output_data),
                    }
                    self.request.request_runtime_ctx.set("response:type", "JSON")
                else:
                    prompt_dict["[OUTPUT REQUIREMENT]"] = str(prompt_output_data)
            prompt_text = to_prompt_structure(prompt_dict, end="[OUTPUT]:\n")
        request_messages.append({"role": "user", "content": [{"type": "text", "text": prompt_text}]})
        return request_messages

    def construct_completion_prompt(self):
        # - init prompt
        completion_prompt = ""
        # - general instruction
        general_instruction_data = self.request.request_runtime_ctx.get_trace_back("prompt.general")
        if general_instruction_data:
            completion_prompt += f"[GENERAL INSTRUCTION]\n{to_instruction(general_instruction_data)}\n"
        # - role
        role_data = self.request.request_runtime_ctx.get_trace_back("prompt.role")
        if role_data:
            completion_prompt += f"[ROLE SETTINGS]\n{to_instruction(role_data)}\n"
        # - user info
        user_info_data = self.request.request_runtime_ctx.get_trace_back("prompt.user_info")
        if user_info_data:
            completion_prompt += f"[ABOUT USER]\n{to_instruction(user_info_data)}\n"
        # - abstract
        headline_data = self.request.request_runtime_ctx.get_trace_back("prompt.abstract")
        if headline_data:
            completion_prompt += f"[ABSTRACT]\n{to_instruction(headline_data)}\n"
        # - main prompt
        chat_history_data = self.request.request_runtime_ctx.get_trace_back("prompt.chat_history")
        prompt_input_data = self.request.request_runtime_ctx.get_trace_back("prompt.input")
        prompt_info_data = self.request.request_runtime_ctx.get_trace_back("prompt.info")
        prompt_instruct_data = self.request.request_runtime_ctx.get_trace_back("prompt.instruct")
        prompt_output_data = self.request.request_runtime_ctx.get_trace_back("prompt.output")
        if not prompt_input_data and not prompt_info_data and not prompt_instruct_data and not prompt_output_data and not chat_history_data:
            raise Exception("[Request] Missing 'prompt.chat_history', 'prompt.input', 'prompt.info', 'prompt.instruct', 'prompt.output' in request_runtime_ctx. At least set value to one of them.")
        # --- only input
        if prompt_input_data and not chat_history_data and not prompt_info_data and not prompt_instruct_data and not prompt_output_data:
            if completion_prompt == "":
                completion_prompt += to_instruction(prompt_input_data)
            else:
                completion_prompt += f"[OUTPUT]\n{to_instruction(prompt_input_data)}"
        # --- construct prompt
        else:
            # - with output format
            if isinstance(prompt_output_data, (dict, list, set)):
                prompt_dict = {}
                if chat_history_data:
                    prompt_dict["[HISTORY LOGS]"] = ""
                    for message in chat_history_data:
                        prompt_dict["[HISTORY LOGS]"] += f"{message['role']}: {message['content']}\n"
                if prompt_input_data:
                    prompt_dict["[INPUT]"] = to_instruction(prompt_input_data)
                if prompt_info_data:
                    prompt_dict["[HELPFUL INFORMATION]"] = str(prompt_info_data)
                if prompt_instruct_data:
                    prompt_dict["[INSTRUCTION]"] = to_instruction(prompt_instruct_data)
                if prompt_output_data:
                    prompt_dict["[OUTPUT REQUIREMENT]"] = {
                        "TYPE": "JSON can be parsed in Python",
                        "FORMAT": to_json_desc(prompt_output_data),
                    }
                    self.request.request_runtime_ctx.set("response:type", "JSON")
                completion_prompt += to_prompt_structure(
                    prompt_dict,
                    end="[OUTPUT]:\nassistant: "
                        if chat_history_data and len(chat_history_data) > 0
                        else "[OUTPUT]:\n"
                )
            # without output format
            else:
                if prompt_info_data:
                    completion_prompt += f"[HELPFUL INFORMATION]\n{str(prompt_info_data)}\n"
                if prompt_instruct_data:
                    completion_prompt += f"[INSTRUCTION]\n{to_instruction(prompt_instruct_data)}\n"
                if prompt_output_data:
                    completion_prompt += f"[OUTPUT REQUIREMENT]\n{str(prompt_output_data)}\n"
                completion_prompt += "[OUTPUT]\n"
                # chat completion
                if chat_history_data:
                    for message in chat_history_data:
                        completion_prompt += f"{message['role']}: {message['content']}\n"
                    if prompt_input_data:
                        completion_prompt += f"user: {to_instruction(prompt_input_data)}\n"
                    completion_prompt += "assistant: "
                # text completion
                else:
                    if prompt_input_data:
                        completion_prompt += prompt_input_data
        return completion_prompt

    def generate_request_data(self):
        # collect options
        options = self.model_settings.get_trace_back("options", {})
        for key, value in self.default_options.items():
            if key not in options:
                options.update({key: value})
        
        # Get the model name from settings
        model = self.model_settings.get_trace_back("model")
        if not model:
            raise Exception("Model name must be specified in settings for LiteLLM")
            
        if self.request_type == "chat":
            return {
                "model": model,
                "messages": format_request_messages(self.construct_request_messages(), self.model_settings),
                **options,
            }
        elif self.request_type == "completions":
            return {
                "model": model,
                "prompt": self.construct_completion_prompt(),
                **options,
            }
        else:
            raise Exception(f"Can not support request type: {self.request_type}.")

    async def request_model(self, request_data: dict):
        # Handle API key and other settings
        api_key = self.model_settings.get_trace_back("auth.api_key")
        if api_key:
            request_data["api_key"] = api_key
        
        base_url = self.model_settings.get_trace_back("url")
        if base_url:
            request_data["api_base"] = base_url

        # Handle proxy settings if any
        proxy = self.request.settings.get_trace_back("proxy")
        if proxy:
            request_data["proxy"] = proxy

        # Make the request using litellm
        stream = await acompletion(**request_data)
        return stream

    async def broadcast_response(self, response_generator):
        if self.request_type == "chat":
            response_message = {}
            async for part in response_generator:
                delta = dict(part.choices[0].delta)
                for key, value in delta.items():
                    if key not in response_message:
                        response_message[key] = value or ""
                    else:
                        response_message[key] += value or ""
                yield({"event": "response:delta_origin", "data": part})
                yield({"event": "response:delta", "data": part.choices[0].delta.content or ""})
            yield({"event": "response:done_origin", "data": response_message})
            yield({"event": "response:done", "data": response_message["content"]})
        elif self.request_type == "completions":
            response_message = {}
            async for part in response_generator:
                if "choices" in dir(part) and isinstance(part.choices, list) and len(part.choices) > 0:
                    delta = dict(part.choices[0])
                    for key, value in delta.items():
                        if key not in response_message and value is not None:
                            response_message[key] = value
                        elif key in response_message and value is not None:
                            response_message[key] += value
                    yield({"event": "response:delta_origin", "data": delta})
                    yield({"event": "response:delta", "data": delta["text"] or ""})
                else:
                    if self.request.settings.get_trace_back("is_debug"):
                        print(f"[Request] Server Response Message: {str(dict(part))}")
                yield({"event": "response:delta_origin", "data": part})
            yield({"event": "response:done_origin", "data": response_message})
            yield({"event": "response:done", "data": response_message["text"]})

    def export(self):
        return {
            "generate_request_data": self.generate_request_data,
            "request_model": self.request_model,
            "broadcast_response": self.broadcast_response,
        }

def export():
    return ("LiteLLM", LiteLLM) 