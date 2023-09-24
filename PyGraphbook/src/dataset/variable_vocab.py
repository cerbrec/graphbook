import os
import json
from src import graph_util

from typing import List, Mapping, Tuple


def get_all_vocab(operations: List[graph_util.Operation]) -> Mapping[Tuple[str, bool, str], int]:
    """ Get all vocabulary.  Vocab is a mapping from (operation_name, is_input, variable_name) to index. """

    vocab = {}
    for operation in operations:
        for input in operation.inputs:
            vocab[(operation.primitive_name, True, input.primitive_name)] = len(vocab)
        for output in operation.outputs:
            vocab[(operation.primitive_name, False, output.primitive_name)] = len(vocab)

    return vocab


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


def add_static_tensor_vocab(vocab: Mapping[Tuple[str, bool, str], int]) -> None:
    """ Add static tensor vocabulary.

    For every operation input that is not a link but is supplied with static data (i.e., bootstrapped data),
    we assign this input as being supplied data from a "magic variable" called static_tensor. These are special variables
    that correspond to static data that can be bootstrapped, so it has to have less than 3 dimensions and can be
    either INTEGER, TEXT, DECIMAL, or BOOLEAN.
    """

    data_types = [graph_util.DataType.INTEGER, graph_util.DataType.TEXT, graph_util.DataType.DECIMAL, graph_util.DataType.BOOLEAN]
    num_dims = ["0", "1", "2"]

    for data_type in data_types:
        for num_dim in num_dims:
            vocab[(str(data_type), False, num_dim)] = len(vocab)


def create_vocab() -> Mapping[Tuple[str, bool, str], int]:
    """ Create vocabulary. """

    primitives_folder = os.getcwd() + "/../compute_operations/"

    graphs = collect_operations(primitives_folder)
    vocab_map = get_all_vocab(graphs)
    print(f"Vocab length so far: {len(vocab_map)}")
    add_static_tensor_vocab(vocab_map)
    print(f"Vocab length with static tensors: {len(vocab_map)}")
    return vocab_map