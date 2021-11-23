class DictTreeNode:
    """
    Define a tree data structure, whose children are encoded in a dictionary
    and whose value at each node is also encoded in a dictionary.
    Instantiate a tree by its root node.
    """
    def __init__(self, dict_value, parent=None, children=None):
        self._dict = dict_value
        self.parent = parent
        if children is None:
            self.children = {}
    
    def __len__(self):
        return len(self.children)
    
    def __getitem__(self, key):
        return self._dict[key]
    
    def __setitem__(self, key, value):
        self._dict[key] = value
    
    def get(self, key, default=None):
        return self._dict.get(key, default)
    
    def add_children_values_dict(self, children_values_dict):
        for key, value in children_values_dict.items():
            self.children[key] = DictTreeNode(value, parent=self)
        
    def isroot(self):
        return self.parent is None
        
    def isleaf(self):
        return len(self.children) == 0
    
    def __repr__(self):
        return f'{self._dict} {list(self.children.keys())}'