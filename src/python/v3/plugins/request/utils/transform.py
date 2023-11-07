import yaml

def to_prompt_structure(prompt_dict: dict, layer_count: int=1, end: str=""):
    prompt = ""
    for key, content in prompt_dict.items():
        prompt += f"{ '#' * layer_count } { key }:\n"
        if isinstance(content, dict):
            prompt += to_prompt_structure(content, layer_count + 1) + '\n'
        else:
            prompt += str(content) + '\n'
        if layer_count == 1:
            prompt += "\n"
    if layer_count == 1:
        prompt += end
    return prompt

def to_instruction(origin):
    if origin == None:
        return None
    elif isinstance(origin, (list, tuple, set, dict)):
        return yaml.dump(origin, allow_unicode=True, sort_keys=False)
    else:
        return str(origin)

def to_json_desc(origin, layer_count = 0):
    if isinstance(origin, dict):
        json_string = ""
        if layer_count > 0:
            json_string += "\n"
        json_string += ("\t" * layer_count) + "{\n"
        for key, value in origin.items():
            json_string += ("\t" * (layer_count + 1)) + "\"" + key + "\": " + to_json_desc(value, layer_count + 1) + "\n"
        if layer_count > 0:
            json_string += ("\t" * (layer_count + 1)) + "},"
        else:
            json_string += "}"
        return json_string
    elif isinstance(origin, (list, set)):
        json_string = ""
        if layer_count > 0:
            json_string += "\n"
        json_string += ("\t" * layer_count) + "[\n"
        for item in origin:
            json_string += ("\t" * (layer_count + 1)) + to_json_desc(item, layer_count + 1) + ",\n"
        json_string += ("\t" * (layer_count + 1)) + "...\n"
        if layer_count > 0:
            json_string += ("\t" * layer_count) + "],"
        else:
            json_string += "]"
        return json_string
    elif isinstance(origin, tuple):
        json_string = f"<{ origin[0] }>,"
        if len(origin) >= 2:
            json_string += f"//{ origin[1] }"
        return json_string
    else:
        return str(origin)

def find_json(origin: str):
    stack = []
    result = ""
    found_open_bracket = False

    for char in origin:
        if char not in ("\t","\n"):
            if found_open_bracket:
                result += char

            if char == '{':
                found_open_bracket = True
                result += char
                stack.append(char)
            elif char == '}':
                if found_open_bracket:
                    stack.pop()
                    if not stack:
                        break

    if len(result) >= 2 and result[-2] == ",":
        result = result[:-2] + "}"
    return result if stack == [] else None