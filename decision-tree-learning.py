import csv
import math
from collections import Counter
from graphviz import Digraph

def read_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = [row for row in reader]
    return headers, data


def entropy(data):
    label_count = Counter([row[-1] for row in data])
    total = len(data)
    return -sum([(count / total) * math.log2(count / total) for count in label_count.values()])


def information_gain(data, attribute, attribute_values):
    initial_entropy = entropy(data)
    total = len(data)
    weighted_entropy = sum([(len(subset) / total) * entropy(subset) for subset in attribute_values.values()])
    return initial_entropy - weighted_entropy


def split_data(data, attribute, attribute_values):
    return {value: [row for row in data if row[attribute] == value] for value in attribute_values}

gaindict = {}
gains = []
def importance(data, headers):
    gains = []
    for i, header in enumerate(headers[:-1]):
        attribute_values = {row[i] for row in data}
        split = {value: [row for row in data if row[i] == value] for value in attribute_values}
        gain = information_gain(data, i, split)
        gains.append((i, gain))
    return max(gains, key=lambda x: x[1])


def decision_tree_learning(data, headers, parent_data=None):
    if len(set([row[-1] for row in data])) == 1:
        return data[0][-1]

    if len(headers) == 0:
        return Counter([row[-1] for row in parent_data]).most_common(1)[0][0]

    attribute, gain = importance(data, headers)
    tree = {headers[attribute]: {}}
    attribute_values = {row[attribute] for row in data}
    split = split_data(data, attribute, attribute_values)

    gaindict[headers[attribute]] = gain
    print(f"Parent node: {headers[attribute]}, Information Gain: {gain:.4f}")

    remaining_headers = headers[:attribute] + headers[attribute + 1:]

    for value, subset in split.items():
        subtree = decision_tree_learning(subset, remaining_headers, data)
        tree[headers[attribute]][value] = subtree

    return tree

def print_decision_tree(tree, graph, parent=None, edge_label=None):
    if not isinstance(tree, dict):
        leaf_node = f"{tree}"
        graph.node(leaf_node, leaf_node, shape="ellipse", style="filled", fillcolor="lightblue")
        graph.edge(parent, leaf_node, label=edge_label)
        return

    for attribute, subtree in tree.items():
        
        graph.node(attribute, attribute + "?" + " (IG: " + str(round(gaindict[attribute],2)) + ")")
        if parent is None:
            for value, child in subtree.items():
                print_decision_tree(child, graph, attribute, value)
        else:
            graph.edge(parent, attribute, label=edge_label)
            for value, child in subtree.items():
                print_decision_tree(child, graph, attribute, value)

def main():
    headers, data = read_csv("data.csv")
    tree = decision_tree_learning(data, headers)
    graph = Digraph("Decision Tree",format="png")
    print_decision_tree(tree, graph)
    graph.view()

if __name__ == "__main__":
    main()
