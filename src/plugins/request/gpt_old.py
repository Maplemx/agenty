import asyncio
import aiohttp
import json
from .utils import to_instruction, to_json_desc

def construct_prompt_from_dict(prompt_dict, layer_count = 1):
    prompt = ""
    for key, content in prompt_dict.items():
        prompt += f"{ '#' * layer_count } { key }:\n"
        if isinstance(content, dict):
            prompt += construct_prompt_from_dict(content, layer_count + 1) + '\n'
        else:
            prompt += str(content) + '\n'
        if layer_count == 1:
            prompt += "\n"
    if layer_count == 1:
        prompt += "# [OUTPUT]:\n"
    return prompt

def construct_request_messages(request_runtime_ctx):
    #init request messages
    request_messages = []
    # - system message
    system_data = request_runtime_ctx.get("system")
    if system_data:
        request_messages.append({ "role": "system", "content": to_instruction(system_data) })
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
        request_messages.append({ "role": "user", "content": construct_prompt_from_dict(prompt_dict) })
    return request_messages

def generate_request_data(get_settings, request_runtime_ctx):
    #default request data
    request_data = {
        "url": "https://api.openai.com/v1/chat/completions",
        "headers": {
            "Authorization": None,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        "data": {
            "model": "gpt-3.5-turbo-1106",
            "temperature": 1,
            "stream": True
        },
        "proxy": None
    }
    #model settings
    model_settings_data = get_settings("model_settings")
    if "auth" in model_settings_data and "api_key" in model_settings_data["auth"]:
        request_data["headers"]["Authorization"] = f"Bearer { model_settings_data['auth']['api_key'] }"
    else:
        raise Exception("[GPT Request] Missing 'model_settings.auth.api_key' in request runtime_ctx.")
    if "url" in model_settings_data and isinstance(model_settings_data["url"], str):
        request_data["url"] = model_settings_data["url"]
    if "options" in model_settings_data and isinstance(model_settings_data["options"], dict):
        request_data["data"].update(model_settings_data["options"])
    #proxy
    request_data["proxy"] = get_settings("proxy")
    #construct request messages
    request_messages = construct_request_messages(request_runtime_ctx)
    if request_runtime_ctx.get("response:type") == "JSON"\
    and request_data["data"]["model"] in ("gpt-4-1106-preview", "gpt-3.5-turbo-1106"):
        request_data["data"].update({ "response_format": { "type": "json_object" } })
    #transform request messages to model required format
        #nah, no need for OpenAI GPT since it is the base format
    request_data["data"].update({ "messages": request_messages })
    return request_data

async def request_model(request_data):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            request_data["url"],
            headers = request_data["headers"],
            json = request_data["data"],
            proxy = request_data["proxy"],
        ) as response:
            async for chunk in response.content.iter_chunks():
                items = chunk[0].decode("utf-8").split("\n\n")
                items = [item for item in items if item != '']
                for item in items:
                    if item[:6] == 'data: ':
                        delta = item[6:]
                        if delta == '[DONE]':
                            yield { "is_done": True }
                            break
                        else:
                            yield { "is_done": False, "delta" : json.loads(delta) }
                    else:
                        raise Exception(f"[GPT Request] Error occurred when request API:\n { str(item) }")
                
def content_mapping(event_type: str, origin_data: dict):
    if event_type == "delta":
        if "delta" in origin_data["choices"][0] and "content" in origin_data["choices"][0]["delta"]:
            return origin_data["choices"][0]["delta"]["content"]
        else:
            return None
    elif event_type == "done":
        return origin_data["choices"][0]["content"]

async def broadcast_response(response_generater):
    response_message = {}
    full_response_message = {}
    async for data in response_generater:
        if not data["is_done"]:
            if "choices" not in data["delta"]:
                raise Exception(f"[GPT Request] Error occurred when request API:\n { str(data) }")
            else:
                delta_content = data["delta"]["choices"][0]
                if "delta" in delta_content:
                    full_response_message.update(data["delta"])
                    for key, value in delta_content["delta"].items():
                        value = response_message[key] + value if key in response_message else value
                        response_message.update({ key: value })
                    yield { "event": "response:delta_origin", "data": data["delta"] }
                    response_content = content_mapping("delta", data["delta"])
                    if response_content != None and response_content != '':
                        yield { "event": "response:delta", "data": response_content }
        else:
            del full_response_message["choices"][0]["delta"]
            full_response_message["choices"][0].update(response_message)
            yield { "event": "response:done_origin", "data": full_response_message }
            yield { "event": "response:done", "data": content_mapping("done", full_response_message) }

def export():
    return ('gpt', {
        "generate_request_data": generate_request_data,
        "request_model": request_model,
        "broadcast_response": broadcast_response,
    })