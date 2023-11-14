import Agently
# turn on debug mode to watch processing logs
agent_factory = Agently.AgentFactory(is_debug = True)

# This time let's try ERNIE(文心大模型)-4 to drive function calling
# You can switch to any model that Agently support
agent_factory\
    .set_settings("current_model", "ERNIE")\
    .set_settings("model.ERNIE.model_name", "ERNIE")\
    .set_settings("model.ERNIE.auth", { "aistudio": "<YOUR-ERNIE-ACCESS-TOKEN>" })

# First let's create a worker agent
worker = agent_factory.create_agent()

# Prepare tool descriptions (of course you can load them from your system storage)
tools = {
    "weather_report": {
        "desc": "get weather report for the present time",
        "input_requirement": {
            "location": ("String", "your location")
        },
        "func": lambda **kwargs: print("The weather is sunny right now.\n", kwargs)
    },
    "weather_forecast": {
        "desc": "get weather forecast for the next 2-24 hours.",
        "input_requirement": {
            "location": ("String", "your location"),
        },
        "func": lambda **kwargs: print("There'll be raining 3 hours later.\n", kwargs)
    },
    "file_browser": {
        "desc": "Browse files that are given to.",
        "input_requirement": {
            "file_path": ("String", "File path that to be browsed."),
            "chunk_num": ("Number", "How many chunks to be output?"),
            "need_summarize": ("Boolean", "Do user need a summarize about the file?")
        },
        "func": lambda **kwargs: print("File browse work done.\n", kwargs)
    },
}

# Then let the worker agent to decide when and how to call the tools
def call_tools(natural_language_input):
    #step 1. confirm tools to be used
    tools_desc = []
    for tool_name, tool_info in tools.items():
        tools_desc.append({ "name": tool_name, "desc": tool_info["desc"] })
    tools_to_be_used = worker\
        .input({
            "input": natural_language_input,
            "tools": str(tools_desc)
        })\
        .output([("String", "Tool name in {{input.tools}} to response {{input}}'s requirement.")])\
        .start()
    #step 2. genrate parameters in one time
    tool_params = {}
    for tool_name in tools_to_be_used:
        tool_params[tool_name] = tools[tool_name]["input_requirement"]
    call_parameters = worker\
        .input({
            "input": natural_language_input,
        })\
        .output(tool_params)\
        .start()
    #step 3. call functions once a time and see the outcomes
    for tool_name, call_parameter in call_parameters.items():
        tools[tool_name]["func"](**call_parameter)
call_tools("Browse ./readme.pdf for me and chunk to 3 pieces without summarize and check Beijing's next 24 hours weather for me.")