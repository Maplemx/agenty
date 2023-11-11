#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/11 17:07
# @Author  : yongjie.su
# @File    : minimax_demo.py
# @Software: PyCharm
from src import AgentFactory

group_id = ""
api_key = ""
base_url = "https://api.minimax.chat/v1/text/chatcompletion"
# 创建一个Agent工厂实例
# 通过is_debug=True开启研发调试模式，获得更透明的过程信息
agent_factory = AgentFactory(is_debug=True)

# 设置工厂的配置项（这些配置项将被工厂生产的所有Agent继承）
agent_factory \
    .set_settings("model_settings.model_name", "MiniMax") \
    .set_settings("model_settings.auth", {"api_key": api_key, "group_id": group_id}) \
    .set_settings("model_settings.options", {})
# 创建一个Agent实例
agent = agent_factory.create_agent()

agent.request.request_runtime_ctx.set("request_type", "chat")
agent.request.request_runtime_ctx.set("prompt", "你是一个专家。")


# 利用agent实例提供的丰富能力接口进行诉求表达
# 链式表达的形式，让诉求表达结构清晰有条理
result = agent \
    .input("给我介绍一下北京周边的景点。") \
    .start()
# 并在最终获得结构化的数据返回结果
print(result)
