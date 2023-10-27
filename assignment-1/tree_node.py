class TreeNode:
    def __init__(
        self, attr: str, isRoot: bool, value: bool = False, common_value=False
    ):
        self.attr = attr  # Nombre del atributo a evaluar
        self.common_value = common_value
        self.isRoot = isRoot  # Nodo raiz o hoja
        self.children = None  # Diccionario con valor -> nodo
        if not isRoot:
            self.value = value  # Para casos hoja el valor objetivo

    def append_node(self, branch_value, new_node):
        if not self.children:
            self.children = {branch_value: new_node}
        else:
            self.children[branch_value] = new_node

    def print_tree(self, depth=0):
        spacer = "| "
        if not self.isRoot:
            print(spacer * (depth), end="")
            print(self.value)
        else:
            print(spacer * (depth), end="")
            print(self.attr)
            for branch_value, child_node in self.children.items():
                print(spacer * (depth + 1), end="")
                print(branch_value)
                child_node.print_tree(depth + 2)
