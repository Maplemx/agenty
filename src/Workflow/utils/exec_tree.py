from ..Schema import Schema


def prepare_data(schema: Schema):
    edges_target_map = {}
    edges_source_map = {}
    edges = schema.edges

    # Step 1. 处理得到原始数据
    for edge in edges:
        edges_target_map.setdefault(edge['source'], []).append(edge)
        edges_source_map.setdefault(edge['target'], []).append(edge)

    # Step 2. 进一步处理数据
    for edge in edges:
        target = edge['target']
        target_handle = edge['target_handle']
        target_node = schema.get_chunk(target)
        points_desc = target_node['points']
        target_handle_desc = next((point for point in points_desc.get('inputs', []) if point['handle'] == target_handle), None)
        if target_handle_desc:
            edge['target_handle_title'] = target_handle_desc['title']

    full_logic_tree = []

    def build_logic_nodes(current_item, parent=None):
        targets = edges_target_map.get(current_item['id'], [])
        branches = []
        current_branch_recorder = {}
        for target in targets:
            node_id = target['target']
            if node_id in current_branch_recorder:
                current_branch_recorder[node_id]['sources']['handlers'].append(target['source_handle'])
            else:
                target_chunk = (schema.get_chunk(node_id) or {})
                branch = {
                    'type': target_chunk.get('type'),
                    'id': node_id,
                    'data': target_chunk,
                    'pointType': 'end' if not edges_target_map.get(node_id) else 'normal',
                    'sources': {
                        'id': target['source'],
                        'handlers': [target['source_handle']]
                    }
                }
                current_branch_recorder[node_id] = branch
                branch['branches'] = build_logic_nodes(branch, current_item)
                branches.append(branch)
        return branches

    for node in schema.chunks:
        if not edges_source_map.get(node['id'], []):
            root_logic_node = {
                'type': 'node',
                'id': node['id'],
                'data': node,
                'pointType': 'start',
                'prev': None
            }
            root_logic_node['branches'] = build_logic_nodes(root_logic_node)
            full_logic_tree.append(root_logic_node)

    return {
        'edges_source_map': edges_source_map,
        'full_logic_tree': full_logic_tree
    }


def generate_exec_tree(schema: Schema):
    prepared_data = prepare_data(schema)

    def create_exec_tree_node(node):
        return {
            'id': node['id'],
            'deps': [
                {
                    'id': dep['source'],
                    'handler': dep['source_handle'],
                    'target_handler': dep['target_handle']
                } for dep in prepared_data['edges_source_map'].get(node['id'], [])
            ],
            'data': node['data'],
            'branches': []
        }

    exec_tree = []

    def logic_tree_walker(nodes, parent_collections):
        for node in nodes:
            exec_tree_node = create_exec_tree_node(node)
            parent_collections.append(exec_tree_node)
            if node.get('branches') and node['branches']:
                logic_tree_walker(node['branches'], exec_tree_node['branches'])

    logic_tree_walker(prepared_data['full_logic_tree'], exec_tree)
    return exec_tree
