
class DecisionTreeNode:
    def __init__(self, attribute=None, label=None, depth=0, parent=None):
        self.attribute = attribute
        self.info_gain = 0
        self.children = {}
        self.label = label
        self.depth = depth
        self.parent = parent

    def add_child(self, value, node):
        node.parent = self
        node.depth = self.depth + 1
        self.children[value] = node