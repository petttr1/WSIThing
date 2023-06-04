class Storage:
    def __init__(self):
        self.storage = {}

    def add_to_storage(self, item, key):
        self.storage[key] = item
        return key

    def get(self, key):
        return self.storage[key]
