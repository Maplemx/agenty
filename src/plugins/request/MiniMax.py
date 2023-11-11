#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 15:55
# @Author  : yongjie.su
# @File    : MiniMax.py
# @Software: PyCharm
import json
import aiohttp
from .utils import RequestABC

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
        self.request_type = self.request.request_runtime_ctx.get("request_type", "chat")
        if self.request_type is None:
            self.request_type = "chat"

    def generate_request_data(self, get_settings, request_runtime_ctx):
        options = get_settings("model_settings.options", {})
        if not isinstance(options, dict):
            raise ValueError("The value type of 'model_settings.options' must be a dict.")
        auth = get_settings("model_settings.auth", {})
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
        text = self.request.request_runtime_ctx.get("prompt_input")
        if self.request_type.lower() == "chat_pro":
            raise Exception(f"This type '{self.request_type}' is currently not supported.")
        elif self.request_type.lower() == "chat":
            base_url = options.get('base_url', minimax_chat.get('default_url'))
            body.update({
                "model": options.get('model', minimax_chat.get('default_model_name')),
                "prompt": prompt,
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
        else:
            raise Exception(f"This type '{self.request_type}' is currently not supported.")
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
