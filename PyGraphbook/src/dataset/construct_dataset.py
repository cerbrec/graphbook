""" Construct dataset from Graphbook graph. """

import json
import os
from pydantic import BaseModel, Field
from typing import List, Mapping, Tuple

from src import graph_util

def get_all_vocab(operations: List[graph_util.Operation]) -> Mapping[Tuple[str, bool, str], int]:
    """ Get all vocabulary.  Vocab is a mapping from (operation_name, is_input, variable_name) to index. """

    vocab = {}
    for operation in operations:
        for input in operation.inputs:
            vocab[(operation.primitive_name, True, input.primitive_name)] = len(vocab)
        for output in operation.outputs:
            vocab[(operation.primitive_name, False, output.primitive_name)] = len(vocab)

    return vocab

def add_static_tensor_vocab(vocab: Mapping[Tuple[str, bool, str], int]) -> None:
    """ Add static tensor vocabulary.

    For every operation input that is not a link but is supplied with static data (i.e., bootstrapped data),
    we assign this input as being supplied data from a "magic" called static_tensor. These are special variables
    that correspond to static data that can be bootstrapped, so it has to have less than 3 dimensions and can be
    either INTEGER, TEXT, DECIMAL, or BOOLEAN.
    """

    data_types = [graph_util.DataType.INTEGER, graph_util.DataType.TEXT, graph_util.DataType.DECIMAL, graph_util.DataType.BOOLEAN]
    num_dims = [0, 1, 2]

    for data_type in data_types:
        for num_dim in num_dims:
            vocab[(data_type, False, num_dim)] = len(vocab)


def collect_operations(folder_with_primitives: str) -> List[graph_util.Operation]:
    """ Collect operations. """

    primitive_operations = []
    for sub_folder in os.listdir(folder_with_primitives):
        # If the sub_folder is a folder
        if os.path.isdir(os.path.join(folder_with_primitives, sub_folder)):
            for file in os.listdir(os.path.join(folder_with_primitives, sub_folder)):
                if file.endswith(".json"):
                    with open(os.path.join(folder_with_primitives, sub_folder, file), "r") as f:
                        graph_obj = graph_util.Operation.model_validate(json.load(f))
                        if graph_obj.type == graph_util.OperationType.PRIMITIVE_OPERATION:
                            primitive_operations.append(graph_obj)

    return primitive_operations


class HierarchicalDataset(BaseModel):
    """ Hierarchical Graph input sequence.

        vocab_ids = [... , ... , ...], -1 for a "composite" vocab.
        adj_matrix = [[... , ... , ...], [... , ... , ...], [... , ... , ...]]
        graph_level_ids = [... , ... , ...]
    """

    name: str = Field(..., description="Name of the Graph.")
    variables: List[List[int]] = Field(default=list(), description="Vocab Ids for Variables")
    adj_matrix: List[List[List[int]]] = Field(default=list(), description="Adjacency Matrix")
    graph_level_ids: List[List[int]] = Field(default=list(), description="Graph Level Ids")


global counter
counter = 0

def _convert_graph_to_dataset(
        dataset: HierarchicalDataset,
        graph: graph_util.Operation,
        vocab: Mapping[Tuple[str, bool, str], int]):

    """ Convert graph to dataset. """

    if graph.type == graph_util.OperationType.CONDITIONAL_OPERATION:
        return

    global counter

    composite_inputs = {}
    composite_outputs = {}

    for i, inp in enumerate(graph.inputs):
        composite_inputs[inp.name] = i

    for i, output in enumerate(graph.outputs):
        composite_outputs[output.name] = i


    var_list = []
    level_list = []
    var_to_id = {}

    for operation in graph.operations:
        if operation.type == graph_util.OperationType.PRIMITIVE_OPERATION:
            # Then get vocab
            for inp in operation.inputs:
                var_list.append(vocab[(operation.primitive_name, True, inp.primitive_name)])
                level_list.append(-1)
                var_to_id[(operation.name, True, inp.name)] = len(var_list) - 1
            for output in operation.outputs:
                var_list.append(vocab[(operation.primitive_name, False, output.primitive_name)])
                level_list.append(-1)
                var_to_id[(operation.name, False, output.name)] = len(var_list) - 1
        elif operation.type == graph_util.OperationType.CONDITIONAL_OPERATION:
            continue
        else:
            counter += 1
            # It's composite
            for i, inp in enumerate(operation.inputs):
                var_list.append(- i)
                level_list.append(counter)
                var_to_id[(operation.name, True, inp.name)] = len(var_list) - 1

            for i, output in enumerate(operation.outputs):
                var_list.append(- i)
                level_list.append(counter)
                var_to_id[(operation.name, False, output.name)] = len(var_list) - 1

            _convert_graph_to_dataset(dataset, operation, vocab)

    pairs = []
    for link in graph.links:
        # Construct adjacency matrix
        if link.source.operation == "this":
            source_id = -100 - composite_inputs[link.source.data]
        else:
            source_id = var_to_id[(link.source.operation, False, link.source.data)]

        if link.sink.operation == "this":
            sink_id = -100 - composite_outputs[link.sink.data]
        else:
            sink_id = var_to_id[(link.sink.operation, True, link.sink.data)]

        pairs.append((source_id, sink_id))

    adj_matrix = []
    for i in range(len(var_list)):
        row = []
        for j in range(len(var_list)):
            if (i, j) in pairs:
                row.append(1)
            else:
                row.append(0)
        adj_matrix.append(row)

    dataset.variables.append(var_list)
    dataset.adj_matrix.append(adj_matrix)
    dataset.graph_level_ids.append(level_list)

def convert_graph_to_dataset(
        graph: graph_util.Operation,
        vocab: Mapping[Tuple[str, bool, str], int]) -> HierarchicalDataset:

    """ Convert graph to dataset. """

    dataset = HierarchicalDataset(name=graph.name)
    _convert_graph_to_dataset(dataset, graph, vocab)
    return dataset


if __name__ == "__main__":
    folders = os.getcwd() + "/../../../compute_operations/"
    graphs = collect_operations(folders)

    vocab_map = get_all_vocab(graphs)

    print(f"Vocab length so far: {len(vocab_map)}")

    add_static_tensor_vocab(vocab_map)

    print(f"Vocab length after adding static tensors: {len(vocab_map)}")

    """
        Each composite graph is a sequence of vocab Ids  and adj matrix
        
        vocab_ids = [... , ... , ...], -1 for a "composite" vocab.
        adj_matrix = [[... , ... , ...], [... , ... , ...], [... , ... , ...]]
        graph_level_ids = [... , ... , ...]
    """

    # graph = graph.read_graphbook_from_file(os.getcwd() + "/../../../nlp_models/transformers/GPT-2.json")
    graph = graph_util.read_graphbook_from_file(os.getcwd() + "/../../../compute_operations/common_layer_operations/Softmax.json")
    # Convert graph to input sequence.

    datasets = []
    datasets.append(convert_graph_to_dataset(graph, vocab_map))

    counter = 0
    graph = graph_util.read_graphbook_from_file(os.getcwd() + "/../../../compute_operations/common_layer_operations/Layer Normalization.json")
    datasets.append(convert_graph_to_dataset(graph, vocab_map))
    print("CHECK")