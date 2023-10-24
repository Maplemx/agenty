import uuid

class IdGenerator(object):
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.namespace_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, self.namespace)

    def create(self):
        individual_id = uuid.uuid4()
        return str(uuid.uuid5(self.namespace_uuid, str(individual_id)))