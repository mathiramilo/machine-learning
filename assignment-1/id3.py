import numpy as np
from numpy import log2
from pandas import DataFrame
from tree_node import TreeNode


class ID3:
    def __init__(self, min_samples_split=0, min_split_gain=0):
        self.root_node = None
        self.min_samples_split = min_samples_split
        self.min_split_gain = min_split_gain
        pass

    def fit(self, X, y):
        self.root_node = self.algorithm(X, y)
        pass

    def predict(self, X: DataFrame):
        res = []
        for _, row in X.iterrows():
            if self.root_node is not None:
                node = self.root_node
                while node.isRoot:
                    value = row[node.attr]
                    if value in node.children.keys():
                        node = node.children[value]
                    else:
                        node = TreeNode(None, False, node.common_value)
                res.append(node.value)
            else:
                res.append(False)
        return res

    def algorithm(self, X: DataFrame, y: DataFrame) -> TreeNode:
        """Implementacion del algoritmo ID3 para construir el arbol de decision.

        Args:
            X (DataFrame): Dataframe que contiene los atributos
            y (Dataframe): Dataframe que contiene el atributo objetivo

        Returns:
            TreeNode: Árbol de decisión
        """

        attr_list = X.columns.tolist()

        # Si todos los registros tienen el mismo valor, entonces se devuelve un nodo hoja con ese valor
        if len(y.unique()) == 1:
            return TreeNode(None, False, y.unique()[0] != 0)

        # Si no quedan atributos para evaluar, devolver un nodo hoja con el valor más común
        if len(attr_list) == 0:
            return TreeNode(None, False, y.value_counts().idxmax() != 0)
    
        # Si el subconjunto de datos es vacío, crear un nodo hoja etiquetado con el valor mas probable.
        if len(X) < self.min_samples_split:
            return TreeNode(None, False, y.value_counts().idxmax() != 0)

        # En caso contrario, encontrar el mejor atributo para dividir el conjunto de datos y crear un nodo con ese atributo
        best_attr = self.find_best_attribute(X, y)
        if (best_attr is None):
            return TreeNode(None, False, y.value_counts().idxmax() != 0)
        
        tree = TreeNode(
            attr=best_attr, isRoot=True, common_value=(y.value_counts().idxmax() != 0)
        )

        for value in X[best_attr].unique():
            # Crear subconjunto de datos que contenga el valor del atributo seleccionado
            subframe = X[X[best_attr] == value]

            

            # En caso contrario, llamar recursivamente a la función con el subconjunto de datos y los atributos restantes y agregar el nodo creado como hijo del nodo actual
            subtree = self.algorithm(
                subframe.drop(
                    columns=[
                        best_attr,
                    ]
                ),
                y.loc[subframe.index],
            )
            tree.append_node(value, subtree)

        return tree

    def find_best_attribute(self, dataframe: DataFrame, target_frame: DataFrame):
        min_entropy = 1
        best_attr = None
        for attr in dataframe.columns:
            tmp_entropy = self.attr_entropy(attr, dataframe, target_frame)
            # print("E[{}]: {}".format(attr, tmp_entropy))
            if tmp_entropy < min_entropy and tmp_entropy <= 1 - self.min_split_gain:
                min_entropy = tmp_entropy
                best_attr = attr
        return best_attr

    def attr_entropy(self, attr: str, dataframe: DataFrame, target_frame: DataFrame):
        unique_values = dataframe[attr].unique()
        total_entropy = 0
        total_count = len(dataframe)
        for value in unique_values:
            subframe = dataframe[dataframe[attr] == value]
            positive_count = len(target_frame.loc[subframe.index][target_frame != 0])
            negative_count = len(target_frame.loc[subframe.index][target_frame == 0])
            total_entropy += (len(subframe) / total_count) * self.entropy(
                positive_count, negative_count
            )
            # print("entropia de {}: {}".format(attr, total_entropy))
        return total_entropy

    def entropy(self, positives: int, negatives: int) -> float:
        total = positives + negatives
        positive_ratio = positives / total if total != 0 else 0
        negative_ratio = negatives / total if total != 0 else 0
        return -positive_ratio * (
            log2(positive_ratio) if positive_ratio != 0 else 0
        ) - negative_ratio * (log2(negative_ratio) if negative_ratio != 0 else 0)
