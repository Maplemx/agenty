import json
from .transform import find_json, to_json_desc

def check_structure(origin: any, compare_target: any, position: str=""):
    errors = []
    if isinstance(origin, dict):
        for key in origin.keys():
            current_position = position + "." + key
            if key not in compare_target:
                errors.append(current_position[1:])
            check_result = check_structure(origin[key], compare_target[key], current_position)
            if check_result != True:
                errors.extend(check_result)
    if isinstance(origin, list):
        for index, item in enumerate(origin):
            current_position = position + "." + str(index)
            if index in enumerate(compare_target):
                check_result = check_structure(item, compare_target[index], current_position)
                if check_result != True:
                    errors.extend(check_result)
    if len(errors) == 0:
        return True
    else:
        return errors

def fix_json_structure(origin: str, input_dict: any, output_dict: any, request: object, *, is_debug: bool=False, errors = list):
    try:
        fixed_result = request\
            .input({
                "this JSON is mean to do": input_dict,
                "expect structure": to_json_desc(output_dict),
                "error JSON String": origin ,
                "error": "structure of {error JSON String} is not the same as {except structure}",
                "error position": errors,
            })\
            .output('FIXED JSON STRING ONLY WITHOUT EXPLANATION that can be parsed by Python')\
            .start()
        fixed_result = json.loads(find_json(fixed_result))
        if is_debug:
            print("[Parse JSON to Dict] Done")
            print("\n--------------------------\n")
        return fixed_result
    except Exception as e:
        raise Exception(f"[Agent Request] Error still occured when try to fix JSON decode error: { str(e) }")

def fix_json_format(origin: str, input_dict: any, output_dict: any, request: object, *, is_debug: bool=False, error: str, position: str):
    try:
        fixed_result = request\
            .input({
                "this JSON is mean to do": input_dict,
                "expect format": to_json_desc(output_dict),
                "error JSON String": origin ,
                "error": error,
                "position": position,
            })\
            .output('FIXED JSON STRING ONLY WITHOUT EXPLANATION that can be parsed by Python')\
            .start()
        fixed_result = json.loads(find_json(fixed_result))
        if is_debug:
            print("[Parse JSON to Dict] Done")
            print("\n--------------------------\n")
        return fixed_result
    except Exception as e:
        raise Exception(f"[Agent Request] Error still occured when try to fix JSON decode error: { str(e) }")

def load_json(origin: str, input_dict: any, output_dict: any, request: object, *, is_debug: bool=False):
    try:
        json_string = find_json(origin)
        parsed_dict = json.loads(json_string)
        check_structure_result = check_structure(parsed_dict, output_dict)
        if check_structure_result != True:
            return fix_json_structure(json_string, input_dict, output_dict, request, is_debug = is_debug, errors = check_structure_result)
        else:
            if is_debug:
                print("[Parse JSON to Dict] Done")
                print("\n--------------------------\n")
            return parsed_dict
    except json.JSONDecodeError as e:
        return fix_json_format(json_string, input_dict, output_dict, request, is_debug = is_debug, error = e.msg, position = e.pos)