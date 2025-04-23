class IDMapper:
    def __init__(self):
        self.parent = {}

    def find(self, id_):
        # If id_ is not in parent, it is its own root
        if id_ not in self.parent:
            self.parent[id_] = id_
        # Path compression
        if self.parent[id_] != id_:
            self.parent[id_] = self.find(self.parent[id_])
        return self.parent[id_]

    def union(self, ids):
        # Merge all ids to the smallest one
        root = min(self.find(id_) for id_ in ids)
        for id_ in ids:
            self.parent[self.find(id_)] = root

    def get_mapping(self):
        # Build final mapping with full resolution
        return {id_: self.find(id_) for id_ in self.parent}