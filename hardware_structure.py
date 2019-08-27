class Hw:
    def __init__(self, index, hw_usage):
        self.index = index
        self.hw_usage = hw_usage
    def __repr__(self):
        return repr((self.index, self.hw_usage))

class Container:
    def __init__(self, container_id , container_size):
        self.container_id = container_id
        self.container_size = container_size
    def __repr__(self):
        return repr((self.container_id, self.container_size))

class Order:
    def __init__(self, container_id , backup_type, backup_order):
        self.container_id = container_id
        self.backup_type = backup_type
        self.backup_order = backup_order
    def __repr__(self):
        return repr((self.container_id, self.backup_type, self.backup_order))

