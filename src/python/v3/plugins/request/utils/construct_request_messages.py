from .transform import to_instruction, to_json_desc

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
                prompt_dict["OUTPUT"] = str(prompt_output_data)
        request_messages.append({ "role": "user", "content": construct_prompt_from_dict(prompt_dict) })
    return request_messages