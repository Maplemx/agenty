import yaml

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

def find_all_jsons(origin: str):
    stage = 1
    json_blocks = []
    block_num = 0
    layer = 0
    skip_next = False
    in_quote = False
    for char in origin:
        if skip_next:
            skip_next = False
            continue
        if stage == 1:
            if char == "\\":
                skip_next = True
                continue
            if char == "[" or char == "{":
                json_blocks.append(char)
                stage = 2
                layer += 1
                continue
        elif stage == 2:
            if char == "\\":
                skip_next = True
                continue
            if char == "\"":
                in_quote = not in_quote
            if char == "[" or char == "{":
                json_blocks[block_num] += char
                layer += 1
            elif char == "]" or char == "}":
                json_blocks[block_num] += char
                layer -= 1
            else:
                if in_quote:
                    json_blocks[block_num] += char
                elif char not in (" ", "\t", "\n"):
                    json_blocks[block_num] += char
            if layer == 0:
                block_num += 1
                stage = 1
    return json_blocks

def find_json(origin: str):
    result = find_all_jsons(origin)
    if len(result) > 0:
        return result[0]
    else:
        return None