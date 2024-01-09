def validate_dict(input_dict, required_keys = []):
    # 检查是否包含所有必需的键
    for key in required_keys:
        if key not in input_dict:
            return {'status': False, 'key': key }
    
    return { 'status': True }