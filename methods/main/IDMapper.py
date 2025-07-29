import numpy as np

class IncrementalFastMapper:
    def __init__(self, initial_size=1024):
        self.parent = np.arange(initial_size, dtype=np.int32)
        self.capacity = initial_size
        self.dirty = set()

    def _ensure_capacity(self, max_id):
        if max_id < self.capacity:
            return

        new_capacity = max_id + 1_000_000
        new_parent = np.arange(new_capacity, dtype=np.int32)
        new_parent[:self.capacity] = self.parent
        self.parent = new_parent
        self.capacity = new_capacity

    def find(self, x):
        self._ensure_capacity(x)

        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        # Path compression
        while x != root:
            p = self.parent[x]
            self.parent[x] = root
            x = p

        return root

    def union(self, ids):
        # Convert to roots
        roots = set(self.find(u) for u in ids)
        if len(roots) < 1:
            return  # All already connected

        rmin = min(roots)
        for r in roots:
            if r != rmin:
                self.parent[r] = rmin
                self.dirty.add(r)  # Mark affected

        for u in ids:
            self.dirty.add(u)  # Track all involved nodes

    def finalize(self):
        for i in self.dirty:
            self.parent[i] = self.find(i)
            
        self.dirty.clear()
        return self.parent