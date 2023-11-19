#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 15:55
# @Author  : yongjie.su
# @File    : MiniMax.py
# @Software: PyCharm
import json
import aiohttp
from loguru import logger
from typing import Union, List
from pydantic import BaseModel, Field

from .utils import RequestABC, to_instruction, to_json_desc, to_prompt_structure
from Agently.utils import RuntimeCtxNamespace


class BaseResp(BaseModel):
    status_code: int = Field(..., description="状态码")
    status_msg: str = Field(..., description="错误详情")


class ChatRoleMeta(BaseModel):
    user_name: str = Field(..., description="用户代称")
    bot_name: str = Field(..., description="ai代称")


class ChatMessage(BaseModel):
    sender_type: str = Field(..., description="发送者")
    text: str = Field(..., description="消息内容")


# chat
class MiniMaxChatInput(BaseModel):
    model: str = Field(..., description="调用的算法模型")
    stream: Union[bool, None] = Field(None, description="是否通过流式分批返回结果")
    use_standard_sse: Union[bool, None] = Field(None, description="是否使用标准SSE格式")
    beam_width: Union[int, None] = Field(None, description="生成多少个结果")
    prompt: str = Field(..., description="请求的模型版本")
    role_meta: ChatRoleMeta = Field(..., description="对话meta信息")
    messages: List[ChatMessage] = Field(..., description="对话内容")
    continue_last_message: Union[bool, None] = Field(None, description="设置当前请求是否为续写模式")
    tokens_to_generate: Union[int, None] = Field(None, description="最大生成token数")
    temperature: Union[float, None] = Field(None, description="较高的值将使输出更加随机")
    top_p: Union[float, None] = Field(None, description="采样方法")
    skip_info_mask: Union[bool, None] = Field(None, description="对输出中易涉及隐私问题的文本信息进行脱敏")


class ChatChoice(BaseModel):
    text: str = Field(..., description="文本结果")
    index: int = Field(..., description="排名")
    finish_reason: str = Field(..., description="结束原因，枚举值")
    delta: str = Field(..., description="回复文本通过delta给出")


class ChatUsage(BaseModel):
    total_tokens: int = Field(..., description="消耗tokens总数，包括输入和输出")


class MiniMaxChatOutput(BaseModel):
    created: int = Field(..., description="请求发起时间")
    model: str = Field(..., description="请求指定的模型")
    reply: str = Field(..., description="推荐的最好结果")
    input_sensitive: bool = Field(..., description="输入命中敏感词")
    input_sensitive_type: int = Field(..., description="输入命中敏感词类型")
    output_sensitive: bool = Field(..., description="输出命中敏感词")
    output_sensitive_type: int = Field(..., description="输出命中敏感词类型")
    choices: List[ChatChoice] = Field(..., description="所有结果")
    usage: ChatUsage = Field(..., description="tokens数使用情况")
    id: str = Field(..., description="调用id")
    base_resp: BaseResp = Field(..., description="错误状态码和详情")


# chat pro
class ChatProMessage(BaseModel):
    sender_type: str = Field(..., description="发送者的类型")
    sender_name: str = Field(..., description="发送者的类型")
    text: str = Field(..., description="消息内容")


class ChatProBotSetting(BaseModel):
    bot_name: str = Field(..., description="具体机器人的名字")
    content: str = Field(..., description="具体机器人的设定")


class ChatProReplyConstraints(BaseModel):
    sender_type: str = Field(..., description="指定回复的角色类型")
    sender_name: str = Field(..., description="指定回复的机器人名称")


class MiniMaxChatProInput(BaseModel):
    model: str = Field(..., description="调用的模型名称")
    stream: Union[bool, None] = Field(None, description="是否通过流式分批返回结果")
    tokens_to_generate: Union[int, None] = Field(None, description="最大生成token数")
    temperature: Union[float, None] = Field(None, description="较高的值将使输出更加随机")
    top_p: Union[float, None] = Field(None, description="请求的模型版本")
    mask_sensitive_info: Union[bool, None] = Field(None, description="对输出中易涉及隐私问题的文本信息进行打码")
    messages: List[ChatProMessage] = Field(..., description="对话内容")
    bot_setting: List[ChatProBotSetting] = Field(..., description="对每一个机器人的设定")
    reply_constraints: ChatProReplyConstraints = Field(..., description="模型回复要求")


class ChatProChoices(BaseModel):
    messages: List[ChatProMessage] = Field(..., description="对话内容")
    index: int = Field(..., description="排名")
    finish_reason: str = Field(..., description="结束原因，枚举值")


class MiniMaxChatProOutput(BaseModel):
    created: int = Field(..., description="请求发起时间, Unixtime, Nanosecond")
    model: str = Field(..., description="请求指定的模型")
    reply: str = Field(..., description="推荐的最好结果")
    input_sensitive: bool = Field(..., description="输入命中敏感词")
    input_sensitive_type: int = Field(..., description="输入命中敏感词类型")
    output_sensitive: bool = Field(..., description="输出命中敏感词")
    output_sensitive_type: int = Field(..., description="输出命中敏感词类型")
    choices: List[ChatProChoices] = Field(..., description="所有结果")
    usage: ChatUsage = Field(..., description="tokens数使用情况")
    id: str = Field(..., description="调用id")
    base_resp: BaseResp = Field(..., description="错误状态码和详情")


class T2aTimberWeights(BaseModel):
    voice_id: str = Field(..., description="请求的音色编号")
    weight: int = Field(..., description="权重")


class MiniMaxT2aInput(BaseModel):
    model: str = Field(..., description="调用的模型名称")
    timber_weights: Union[T2aTimberWeights, None] = Field(None, description="音色相关信息")
    voice_id: Union[str, None] = Field(None, description="请求的音色编号")
    speed: Union[float, None] = Field(None, description="生成声音的语速")
    vol: Union[float, None] = Field(None, description="生成声音的音量")
    pitch: Union[int, None] = Field(None, description="生成声音的语调")
    text: str = Field(..., description="期望生成声音的文本")


class MiniMaxT2aOutput(BaseModel):
    trace_id: str = Field(..., description="生成id")
    base_resp: Union[BaseResp, None] = Field(None, description="错误状态码和详情")


class MiniMaxT2aProInput(BaseModel):
    model: str = Field(..., description="调用的模型名称")
    timber_weights: Union[T2aTimberWeights, None] = Field(None, description="音色相关信息")
    voice_id: Union[str, None] = Field(None, description="请求的音色编号")
    speed: Union[float, None] = Field(None, description="生成声音的语速")
    vol: Union[float, None] = Field(None, description="生成声音的音量")
    pitch: Union[int, None] = Field(None, description="生成声音的语调")
    text: str = Field(..., description="期望生成声音的文本")
    audio_sample_rate: int = Field(..., description="生成声音的采样率")
    bitrate: int = Field(..., description="生成声音的比特率")


class T2aProExtraInfo(BaseModel):
    audio_length: int = Field(..., description="音频时长")
    audio_sample_rate: int = Field(..., description="采样率")
    audio_size: int = Field(..., description="音频大小")
    bitrate: int = Field(..., description="比特率")
    word_count: int = Field(..., description="可读字数")
    invisible_character_ratio: float = Field(..., description="非法字符占比")


class MiniMaxT2aProOutput(BaseModel):
    trace_id: str = Field(..., description="生成id")
    audio_file: str = Field(..., description="合成的音频下载链接")
    subtitle_file: str = Field(..., description="合成的字幕下载链接")
    extra_info: T2aProExtraInfo = Field(..., description="额外信息")
    base_resp: Union[BaseResp, None] = Field(None, description="错误状态码和详情")


class MiniMaxT2aLargeInput(BaseModel):
    pass


class MiniMaxT2aLargeOutput(BaseModel):
    pass


class MiniMaxEmbeddingsInput(BaseModel):
    model: str = Field(..., description="请求的模型版本")
    texts: list = Field(..., description="期望生成向量的文本")
    type: str = Field(..., description="生成向量后的目标使用场景")


class MiniMaxEmbeddingsOutput(BaseModel):
    vectors: list = Field(..., description="请求的模型版本")
    base_resp: BaseResp = Field(..., description="错误状态码和详情")


class MiniMaxHTTPRequestInput(BaseModel):
    base_url: str = Field(..., description="请求链接")
    api_key: str = Field(..., description="api_key")
    group_id: str = Field(..., description="group_id")
    body: Union[dict, None] = Field(None, description="请求数据")


class MiniMaxHTTPRequestModel(BaseModel):
    base_url: str = Field(..., description="请求链接")
    headers: dict = Field(..., description="请求头信息")
    params: dict = Field(..., description="请求路由参数")
    body: Union[dict, None] = Field(None, description="请求数据")


MINIMAX_DEFAULT_URL = "https://api.minimax.chat/v1/text/chatcompletion"
MINIMAX_DEFAULT_SETTINGS = {
    "chat": {
        "default_url": "https://api.minimax.chat/v1/text/chatcompletion",
        "default_model_name": "abab5-chat",
        "default_max_token": 6144,
        "input_schema": MiniMaxChatInput,
        "output_schema": MiniMaxChatOutput,
    },
    "chat_pro": {
        "default_url": "https://api.minimax.chat/v1/text/chatcompletion_pro",
        "default_model_name": "abab5.5-chat",
        "default_max_token": 16384,
        "input_schema": MiniMaxChatProInput,
        "output_schema": MiniMaxChatProOutput,
    },
    "t2a": {
        "default_url": "https://api.minimax.chat/v1/text_to_speech",
        "default_model_name": "speech-01",
        "input_schema": MiniMaxT2aInput,
        "output_schema": MiniMaxT2aOutput,
    },
    "t2a_pro": {
        "default_url": "https://api.minimax.chat/v1/text_to_speech",
        "default_model_name": "speech-01",
        "input_schema": MiniMaxT2aProInput,
        "output_schema": MiniMaxT2aProOutput,
    },
    "t2a_large": {
        "default_url": "",
        "default_model_name": "",
        "input_schema": MiniMaxT2aLargeInput,
        "output_schema": MiniMaxT2aLargeOutput,
    },
    "embeddings": {
        "default_url": "https://api.minimax.chat/v1/embeddings",
        "default_model_name": "embo-01",
        "default_max_token": 4096,
        "input_schema": MiniMaxEmbeddingsInput,
        "output_schema": MiniMaxEmbeddingsOutput,
    }
}


class MiniMaxRequest:
    def __init__(self, request_input: MiniMaxHTTPRequestInput):
        self._api_key = request_input.api_key
        self._group_id = request_input.group_id
        self._base_url = request_input.base_url
        self._body = request_input.body
        self.response = None

    def __base_url(self) -> str:
        if self._base_url is not None:
            return self._base_url
        return MINIMAX_DEFAULT_URL

    def __headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def __params(self) -> dict:
        return {"GroupId": self._group_id}

    def __build_request_model(self) -> MiniMaxHTTPRequestModel:
        request_model = dict(
            base_url=self.__base_url(),
            headers=self.__headers(),
            params=self.__params(),
            body=self._body
        )
        return MiniMaxHTTPRequestModel(**request_model)

    async def async_request(self, timeout=60, **kwargs):
        try:
            if 'timeout' not in kwargs:
                kwargs.update(dict(timeout=timeout))
            request_model: MiniMaxHTTPRequestModel = self.__build_request_model()
            body = request_model.body
            body = {key: value for key, value in body.items() if value is not None}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        url=request_model.base_url,
                        headers=request_model.headers,
                        params=request_model.params,
                        json=body, **kwargs
                ) as response:
                    response = await response.text()
                    self.response = json.loads(response)
        except Exception as err:
            logger.exception(f"aiohttp request failed: {err=}")
        return self.response if self.response is not None else {}

    def finish(self):
        pass


class MiniMax(RequestABC):
    def __init__(self, request):
        self.request = request
        self.model_name = 'MiniMax'
        _request_type = self.request.request_runtime_ctx.get("request_type", "chat")
        self.request_type = _request_type if _request_type is not None else "chat"
        self.model_settings = RuntimeCtxNamespace(f"model.{self.model_name}", self.request.settings)
        self._default_settings = MINIMAX_DEFAULT_SETTINGS.get(self.request_type.lower())
        self._input_schema = self._default_settings.get('input_schema')
        self._output_schema = self._default_settings.get('output_schema')

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
        options = self.model_settings.get_trace_back("options", {})
        if not isinstance(options, dict):
            raise ValueError("The value type of 'model_settings.options' must be a dict.")
        auth = self.model_settings.get_trace_back("auth", {})
        if not isinstance(auth, dict):
            raise ValueError("The value type of 'model_settings.auth' must be a dict.")
        base_url = options.get('base_url', self._default_settings.get('default_url'))
        request_data = {
            "base_url": base_url,
            "group_id": auth.get('group_id'),
            "api_key": auth.get('api_key'),
            "body": None
        }
        model = options.get('model', self._default_settings.get('default_model_name'))

        request_runtime_ctx = self.request.request_runtime_ctx
        stream = request_runtime_ctx.get("stream", False)

        texts = request_runtime_ctx.get("prompt.input")
        texts = texts if isinstance(texts, list) else [texts]
        if self.request_type.lower() == "chat_pro":
            sender_type = request_runtime_ctx.get("sender_type", "USER")
            sender_name = request_runtime_ctx.get("sender_name", "USER")
            messages = [{"sender_type": sender_type,
                         "sender_name": sender_name,
                         "text": text} for text in texts]
            body = self._input_schema(**{
                "model": model,
                "stream": stream,
                "tokens_to_generate": request_runtime_ctx.get('tokens_to_generate', 1024),
                "temperature": request_runtime_ctx.get('temperature', 0.9),
                "top_p": request_runtime_ctx.get('top_p', 0.95),
                "mask_sensitive_info": request_runtime_ctx.get('mask_sensitive_info', True),
                "messages": messages,
                "bot_setting": [
                    {
                        "bot_name": request_runtime_ctx.get('content', 'BOT'),
                        "content": request_runtime_ctx.get('content', '智能助理'),
                    }
                ],
                "reply_constraints": {
                    "sender_type": request_runtime_ctx.get('sender_type', 'BOT'),
                    "sender_name": request_runtime_ctx.get('sender_name', '智能助理')
                }
            })
        elif self.request_type.lower() == "chat":
            # 生成多少个结果；不设置默认为1，最大不超过4。
            beam_width = int(max(1, min(request_runtime_ctx.get('beam_width', 1), 4)))
            prompt = request_runtime_ctx.get('prompt')
            if prompt is None:
                raise ValueError(f"prompt是对话背景、人物或功能设定，必填项。")
            sender_type = request_runtime_ctx.get("sender_type", "USER")
            messages = [{"sender_type": sender_type, "text": text} for text in texts]
            body = self._input_schema(**{
                "model": model,
                "stream": stream,
                "use_standard_sse": request_runtime_ctx.get('use_standard_sse', False),
                "beam_width": beam_width,
                "prompt": prompt,
                "role_meta": {
                    "user_name": request_runtime_ctx.get('use_standard_sse', "USER"),
                    "bot_name": request_runtime_ctx.get('use_standard_sse', "BOT")
                },
                "messages": messages,
                "continue_last_message": request_runtime_ctx.get('continue_last_message', False),
                "tokens_to_generate": request_runtime_ctx.get('tokens_to_generate', 256),
                "temperature": request_runtime_ctx.get('temperature', 0.9),
                "top_p": request_runtime_ctx.get('top_p', 0.95),
                "skip_info_mask": request_runtime_ctx.get('skip_info_mask', False)
            })
        elif self.request_type.lower() == "t2a":
            # 长度限制<500字符
            text = "".join(texts)[:500]
            body = self._input_schema(**{
                "model": model,
                "timber_weights": request_runtime_ctx.get('timber_weights'),
                "voice_id": request_runtime_ctx.get('voice_id', 'female-shaonv'),
                "speed": request_runtime_ctx.get('speed', 1.0),
                "vol": request_runtime_ctx.get('vol', 1.0),
                "pitch": request_runtime_ctx.get('pitch', 0),
                "text": text
            })
        elif self.request_type.lower() == "t2a_pro":
            # 长度限制<35000字符，段落切换用换行符替代
            text = "".join(texts)[:35000]
            body = self._input_schema(**{
                "model": model,
                "timber_weights": request_runtime_ctx.get('timber_weights'),
                "voice_id": request_runtime_ctx.get('voice_id', 'female-shaonv'),
                "speed": request_runtime_ctx.get('speed', 1.0),
                "vol": request_runtime_ctx.get('vol', 1.0),
                "pitch": request_runtime_ctx.get('pitch', 0),
                "text": text,
                "audio_sample_rate": request_runtime_ctx.get('audio_sample_rate', 32000),
                "bitrate": request_runtime_ctx.get('bitrate', 128000)
            })
        elif self.request_type.lower() == "t2a_large":
            raise Exception(f"This type '{self.request_type}' is currently not supported.")
            # 长度限制<10000000字符，格式为zip
            # files = {
            #     'text': open('/Users/minimax/Downloads/21622420008680404.zip', 'rb')
            # }
            # body = self._input_schema(**{
            #     "model": model,
            #     "timber_weights": request_runtime_ctx.get('timber_weights'),
            #     "voice_id": request_runtime_ctx.get('voice_id', 'female-shaonv'),
            #     "speed": request_runtime_ctx.get('speed', 1.0),
            #     "vol": request_runtime_ctx.get('vol', 1.0),
            #     "pitch": request_runtime_ctx.get('pitch', 0),
            #     "audio_sample_rate": request_runtime_ctx.get('audio_sample_rate', 32000),
            #     "bitrate": request_runtime_ctx.get('bitrate', 128000)
            # })
        elif self.request_type.lower() == "embeddings":
            body = self._input_schema(**{
                "model": model,
                "texts": texts,
                "type": request_runtime_ctx.get('type', 'query')
            })
        else:
            raise Exception(f"This type '{self.request_type}' is currently not supported.")
        request_data.update({"body": body.model_dump()})
        return request_data

    async def request_model(self, request_data: dict, timeout=60):
        request_input: MiniMaxHTTPRequestInput = MiniMaxHTTPRequestInput(**request_data)
        model = MiniMaxRequest(request_input)
        result = await model.async_request()
        return self._output_schema(**result)

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
