#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 15:55
# @Author  : yongjie.su
# @File    : MiniMax.py
# @Software: PyCharm
import json
import aiohttp
from .utils import RequestABC, to_instruction, to_json_desc, to_prompt_structure
from Agently.utils import RuntimeCtxNamespace

minimax_chat_pro = {
    "default_url": "https://api.minimax.chat/v1/text/chatcompletion_pro",
    "default_model_name": "abab5.5-chat",
    "default_max_token": 16384
}

minimax_chat = {
    "default_url": "https://api.minimax.chat/v1/text/chatcompletion",
    "default_model_name": "abab5-chat",
    "default_max_token": 6144
}

minimax_t2a = {
    "default_url": "https://api.minimax.chat/v1/text_to_speech",
    "default_model_name": "speech-01"
}

minimax_t2a_pro = {
    "default_url": "https://api.minimax.chat/v1/t2a_pro",
    "default_model_name": "speech-01"
}

minimax_embeddings = {
    "default_url": "https://api.minimax.chat/v1/embeddings",
    "default_model_name": "embo-01",
    "default_max_token": 4096
}


class MiniMax(RequestABC):
    def __init__(self, request):
        self.request = request
        self.model_name = 'MiniMax'
        _request_type = self.request.request_runtime_ctx.get("request_type", "chat")
        self.request_type = _request_type if _request_type is not None else "chat"
        self.model_settings = RuntimeCtxNamespace(f"model.{self.model_name}", self.request.settings)

    def construct_request_messages(self):
        # init request messages
        request_messages = []
        # - general instruction
        general_instruction_data = self.request.request_runtime_ctx.get(
            "prompt.general_instruction")
        if general_instruction_data:
            request_messages.append({"role": "user",
                                     "content": f"[重要指导说明]\n{to_instruction(general_instruction_data)}"})
            request_messages.append({"role": "assistant", "content": "OK"})
        # - role
        role_data = self.request.request_runtime_ctx.get("prompt.role")
        if role_data and self.request_type == "chat":
            request_messages.append(
                {"role": "user", "content": f"[角色及行为设定]\n{to_instruction(role_data)}"})
            request_messages.append({"role": "assistant", "content": "OK"})
        # - user info
        user_info_data = self.request.request_runtime_ctx.get("prompt.user_info")
        if user_info_data and self.request_type == "chat":
            request_messages.append(
                {"role": "user", "content": f"[用户信息]\n{to_instruction(user_info_data)}"})
            request_messages.append({"role": "assistant", "content": "OK"})
        # - headline
        headline_data = self.request.request_runtime_ctx.get("prompt.headline")
        if headline_data:
            request_messages.append(
                {"role": "user", "content": f"[主题及摘要]{to_instruction(headline_data)}"})
            request_messages.append({"role": "assistant", "content": "OK"})
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
            raise Exception(
                "[Request] Missing 'prompt.input', 'prompt.information', 'prompt.instruction', 'prompt.output' in request_runtime_ctx. At least set value to one of them.")
        if prompt_input_data and not prompt_information_data and not prompt_instruction_data and not prompt_output_data:
            request_messages.append({"role": "user", "content": to_instruction(prompt_input_data)})
        # --- construct prompt
        else:
            prompt_dict = {}
            if prompt_input_data:
                prompt_dict["[输入]"] = to_instruction(prompt_input_data)
            if prompt_information_data:
                prompt_dict["[补充信息]"] = to_instruction(prompt_information_data)
            if prompt_instruction_data:
                prompt_dict["[处理规则]"] = to_instruction(prompt_instruction_data)
            if prompt_output_data:
                if isinstance(prompt_output_data, (dict, list, set)):
                    prompt_dict["[输出要求]"] = {
                        "TYPE": "JSON can be parsed in Python",
                        "FORMAT": to_json_desc(prompt_output_data),
                    }
                    self.request.request_runtime_ctx.set("response:type", "JSON")
                else:
                    prompt_dict["[输出要求]"] = str(prompt_output_data)
            request_messages.append(
                {"role": "user", "content": to_prompt_structure(prompt_dict, end="[输出]:\n")})
        return request_messages

    def generate_request_data(self):
        options = self.model_settings.get_trace_back("options")
        if not isinstance(options, dict):
            raise ValueError("The value type of 'model_settings.options' must be a dict.")
        auth = self.model_settings.get_trace_back("auth")
        if not isinstance(auth, dict):
            raise ValueError("The value type of 'model_settings.auth' must be a dict.")
        api_key = auth.get('api_key')
        group_id = auth.get('group_id')
        if not api_key:
            raise ValueError("The value of 'api_key' is invalid.")
        body = {
            "stream": self.request.request_runtime_ctx.get("stream", False)
        }
        prompt = self.request.request_runtime_ctx.get("prompt")
        text = self.request.request_runtime_ctx.get("prompt.input")
        if self.request_type.lower() == "chat_pro":
            base_url = options.get('base_url', minimax_chat_pro.get('default_url'))
            body.update({
                "model": options.get('model', minimax_chat_pro.get('default_model_name')),
                "bot_setting": [
                    {
                        "bot_name": "MM智能助理",
                        "content": "MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。",
                    }
                ],
                "messages": [
                    {
                        "sender_type": "USER",
                        "sender_name": "小明",
                        "text": "帮我用英文翻译下面这句话：我是谁"
                    }
                ],
                "reply_constraints": {
                    "sender_type": "BOT",
                    "sender_name": "MM智能助理"
                },

                "tokens_to_generate": 1034,
                "temperature": 0.01,
                "top_p": 0.95,
            })
        elif self.request_type.lower() == "chat":
            base_url = options.get('base_url', minimax_chat.get('default_url'))
            body.update({
                "model": options.get('model', minimax_chat.get('default_model_name')),
                "prompt": "你是一个专家",
                "role_meta": {
                    "user_name": "USER",
                    "bot_name": "BOT"
                },
                "messages": [{
                    "sender_type": self.request.request_runtime_ctx.get("sender_type", "USER"),
                    "text": text,

                }],
                "temperature": options.get('temperature', 0.5)
            })
        elif self.request_type.lower() == "embeddings":
            base_url = options.get('base_url', minimax_embeddings.get('default_url'))
            body.update({
                "model": options.get('model', minimax_embeddings.get('default_model_name')),
                "texts": text,
                "type": options.get('type', 'query')
            })

        else:
            raise Exception(f"This type '{self.request_type}' is currently not supported.")
        # 统一构造请求数据结构
        request_data = {
            "url": base_url,
            "params": {"GroupId": group_id},
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            "body": body
        }
        return request_data

    async def request_model(self, request_data: dict, timeout=60):
        url = request_data.get('url')
        headers = request_data.get('headers')
        body = request_data.get('body')
        params = request_data.get('params')
        proxy = request_data.get("proxy") if "proxy" in request_data else None
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params, json=body, headers=headers, proxy=proxy,
                                    timeout=timeout) as response:
                response = await response.text()
                response = json.loads(response)
                if response["base_resp"]["status_code"] != 0:
                    raise Exception(response["base_resp"])
                return response.get('choices', [])

    def broadcast_response(self, response_generator):
        response_message = {"role": "assistant", "content": ""}
        full_response_message = {}
        for part in response_generator:
            full_response_message = dict(part)
            delta = part["text"]
            response_message["content"] += delta
            yield ({"event": "response:delta_origin", "data": part})
            yield ({"event": "response:delta", "data": delta})
        full_response_message["result"] = response_message["content"]
        yield ({"event": "response:done_origin", "data": full_response_message})
        yield ({"event": "response:done", "data": response_message["content"]})

    def export(self):
        return {
            "generate_request_data": self.generate_request_data,
            "request_model": self.request_model,
            "broadcast_response": self.broadcast_response,
        }


def export():
    return "MiniMax", MiniMax
