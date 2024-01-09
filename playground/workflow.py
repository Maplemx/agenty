import Agently
agent_factory = Agently.AgentFactory(is_debug=True)
agent_factory\
    .set_settings("current_model", "OpenAI")\
    .set_settings("model.OpenAI.auth", {"api_key": "Your-API-Key-API-KEY"})\
    .set_settings("model.OpenAI.url", "YOUR-BASE-URL-IF-NEEDED")

# 第一步，声明响应函数和初始描述
def workflow_handler(command: str, res: any):
  print(f"Command [{command}]: ", res)


# 第二步，创建 Workflow 实例
workflow = Agently.Workflow(None, workflow_handler, {
    "agent_factory": agent_factory
})

# 实现：判断用户意图是购物还是闲聊，路由到情感专家 or 销售导购
#
#                                     --> 销   售
#     用户输入 --> 意图判断 --> 意图路由               --> 输出
#                                     --> 情感专家
#
chunks = [
    # 初始节点
    {
        "id": "c-global-input",
        "title": "用户的输入",
        "type": "Input",
        "points": {"inputs": [], "outputs": [{"handle": "output", "title": "输出"}]}
    },
    # 输出节点
    {
        "id": "c-global-output",
        "title": "输出",
        "type": "Output",
        "points": {"inputs": [{"handle": "input", "title": "输入"}], "outputs": []}
    },
    # 意图判断节点
    {
        "id": "c-judge",
        "title": "意图判断",
        "type": "AgentRequest",
        "settings": [
            {"name": "Input", "value": {"content": ""}},
            {"name": "Output", "value": {"outputs": [
                {"key": "用户意图", "type": "string", "description": "\"闲聊\" 还是 \"购物\""}]}},
            {"name": "Instruct", "value": {"content": "判断用户意图是“闲聊”还是“购物”"}},
            {"name": "Role", "value": {"角色": {"description": "导购"}}}
        ],
        "points": {
            "inputs": [{"handle": "input", "title": "输入"}], # 入参
            "outputs": [{"handle": "用户意图", "title": "用户意图"}] # 出参，此处 handle 值和 Output 是对应的
        }
    },
    # 意图路由
    {
        "id": "c-judge-router",
        "title": "意图路由",
        "type": "Judge",
        "settings": {
            "conditions": [
                {"id": "res-shopping", "relation": "=", "value": "购物", "value_type": "string"},
                {"id": "others", "relation": "others", "type": "others"}
            ]
        },
        "points": {
            "inputs": [{"handle": "input", "title": "输入"}],
            "outputs": [
                {"handle": "res-shopping", "title": "购物"},
                {"handle": "others", "title": "其它"}
            ]
        }
    },
    {
        "id": "c-sales",
        "title": "销售",
        "type": "AgentRequest",
        "settings": [
            {"name": "Input", "value": {"content": "@$$全局输入 "}},
            {"name": "Output", "value": {"outputs": [
                {"key": "回复", "type": "string", "description": "回答用户的问题"}]}},
            {"name": "Role", "value": {"角色": {"description": "百货超市的销售"}}},
            {"name": "Instruct", "value": {"content": "向用户推销产品，引导用户购买"}}
        ],
        "points": {
            "inputs": [{"handle": "input", "title": "输入"}],
            "outputs": [{"handle": "回复", "title": "回复"}]
        }
    },
    {
        "id": "c-chat",
        "title": "情感专家",
        "type": "AgentRequest",
        "settings": [
            {"name": "Input", "value": {"content": "@$$全局输入 "}},
            {"name": "Output", "value": {"outputs": [
                {"key": "回复", "type": "string", "description": "回答用户的问题"}]}},
            {"name": "Role", "value": {"角色": {"description": "情感专家"}}}
        ],
        "points": {
            "inputs": [{"handle": "input", "title": "输入"}],
            "outputs": [{"handle": "回复", "title": "回复"}]
        }
    }
]
# 添加进 schema 中，你还可以通过 workflow.excutor.regist_excutor 注册自己的节点类型
workflow.schema.add_chunks(chunks)

# 连接节点
workflow.schema.connect_with_edges([
  { 'source': 'c-global-input', 'target': 'c-judge' }, # 连接输入到意图识别
  {'source': 'c-judge', 'source_handle': '用户意图', 'target': 'c-judge-router'}, # 将用户意图结果连接到判断 chunk 上
  { 'source': 'c-judge-router', 'source_handle': 'res-shopping', 'target': 'c-sales' }, # 如果是购物，则路由到销售
  { 'source': 'c-judge-router', 'source_handle': 'others', 'target': 'c-chat' }, # 否则，路由到情感专家
  { 'source': 'c-sales', 'source_handle': '回复', 'target': 'c-global-output' }, # 将销售的回复返回
  { 'source': 'c-chat', 'source_handle': '回复', 'target': 'c-global-output' }, # 将情感专家的回复返回
])

# 第三步，执行
workflow.exec('明天就要放假了，我想买点好吃的')