def find(data_list: list, find_func):
    """
    查找数据
    """
    found_data = None

    for data in data_list:
        if find_func(data):
            found_data = data
            break

    return found_data


def find_by_attr(data_list: list, attr_name: str, attr_val: any):
    """
    查找数据
    """
    found_data = None

    for data in data_list:
        if data.get(attr_name) == attr_val:
            found_data = data
            break

    return found_data


def find_all(data_list: list, find_func):
    """
    查找数据
    """
    found_data = []

    for data in data_list:
        if find_func(data):
            found_data.append(data)
            break

    return found_data


def find_all_by_attr(data_list: list, attr_name: str, attr_val: any):
    """
    查找数据
    """
    found_data = []

    for data in data_list:
        if data.get(attr_name) == attr_val:
            found_data.append(data)
            break

    return found_data

def has_target(data_list: list, judge_func):
    """
    判断数据是否存在
    """
    for data in data_list:
        if judge_func(data):
            return True

    return False


def has_target_by_attr(data_list: list, attr_name: str, attr_val: any):
    """
    判断数据是否存在
    """
    for data in data_list:
        if data.get(attr_name) == attr_val:
            return True

    return False
