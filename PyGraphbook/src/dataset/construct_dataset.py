""" Construct dataset from Graphbook graph. """

import json
import os
import shutil
import argparse
import logging

from pydantic import BaseModel, Field
from typing import List, Mapping, Tuple, Dict

from src import graph_util

global counter
global global_constants

COMPOSITE_INPUT_ID_OFFSET = -10
COMPOSITE_OUTPUT_ID_OFFSET = -50

CONDITIONAL_INPUT_ID_OFFSET = -1000
CONDITIONAL_OUTPUT_ID_OFFSET = -5000

SUB_GRAPH_INPUT_ID_OFFSET = -100
SUB_GRAPH_OUTPUT_ID_OFFSET = -500

PRIMITIVE_LEVEL = -1
DUMMY_LEVEL = -2

BOOTSTRAPPED_DATA_OP_NAME = "static_tensor"

def get_all_vocab(operations: List[graph_util.Operation]) -> Mapping[Tuple[str, bool, str], int]:
    """ Get all vocabulary.  Vocab is a mapping from (operation_name, is_input, variable_name) to index. """

    vocab = {}
    for operation in operations:
        for input in operation.inputs:
            vocab[(operation.primitive_name, True, input.primitive_name)] = len(vocab)
        for output in operation.outputs:
            vocab[(operation.primitive_name, False, output.primitive_name)] = len(vocab)

    return vocab


def add_special_vocab(vocab: List[Tuple[int, Tuple[str, bool, str]]]) -> None:
    """ Add special vocabulary. """

    vocab.append((COMPOSITE_INPUT_ID_OFFSET, ("composite", False, "composite")))
    vocab.append((COMPOSITE_OUTPUT_ID_OFFSET, ("composite", True, "composite")))

    vocab.append((CONDITIONAL_INPUT_ID_OFFSET, ("conditional", False, "conditional")))
    vocab.append((CONDITIONAL_OUTPUT_ID_OFFSET, ("conditional", True, "conditional")))

    vocab.append((SUB_GRAPH_INPUT_ID_OFFSET, ("sub_graph", False, "sub_graph")))
    vocab.append((SUB_GRAPH_OUTPUT_ID_OFFSET, ("sub_graph", True, "sub_graph")))



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


def calculate_longest_sequence(dataset: HierarchicalDataset) -> int:
    """ Calculate longest sequence. """

    return max(len(variables) for variables in dataset.variables)

def _add_static_tensor(
        var_list: List[int],
        vocab: Mapping[Tuple[str, bool, str], int],
        level_list: List[int],
        var_to_id: Dict[Tuple[str, bool, str], int],
        pairs: List[Tuple[int, int]],
        inp: graph_util.Variable) -> None:

    """ Add static tensor. """

    if inp.global_constant:
        inp = global_constants[inp.global_constant]

    shape_length = 0
    if shape_length:
        shape_length = len(inp.shape)

    var_list.append(vocab[(str(inp.type), False, str(shape_length))])

    # Considered primitive because it directly corresponds to a vocab ID
    level_list.append(PRIMITIVE_LEVEL)
    var_to_id[(BOOTSTRAPPED_DATA_OP_NAME, True, inp.name)] = len(var_list) - 1
    pairs.append((len(var_list) - 1, len(var_list) - 2))


def _convert_graph_to_dataset(
        dataset: HierarchicalDataset,
        graph: graph_util.Operation,
        operations: List[graph_util.Operation],
        links: List[graph_util.Link],
        vocab: Mapping[Tuple[str, bool, str], int]) -> int:

    """ Convert graph to dataset. """

    global counter
    counter += 1
    this_level = int(counter)

    logging.debug(f"Graph {graph.name} is at level: {this_level}, row {len(dataset.variables)}")
    # if this_level >= len(dataset.variables):
    # logging.debug(f"Graph {graph.name} is at level: {this_level}")

    var_list = []
    level_list = []
    adj_matrix = []
    pairs = []

    dataset.variables.append(var_list)
    dataset.adj_matrix.append(adj_matrix)
    dataset.graph_level_ids.append(level_list)

    var_to_id = {}

    for i, inp in enumerate(graph.inputs):
        # This is for the subgraph dummy input
        var_list.append(SUB_GRAPH_INPUT_ID_OFFSET - i)
        level_list.append(DUMMY_LEVEL)
        var_to_id[("this", False, inp.name)] = len(var_list) - 1

    if not operations:
        raise ValueError(f"Operations cannot be empty. {graph.name}")
    for operation in operations:
        if operation.type == graph_util.OperationType.PRIMITIVE_OPERATION:
            # Then get vocab
            for inp in operation.inputs:
                var_to_id[(operation.name, True, inp.name)] = len(var_list)

                if inp.flow_state == graph_util.FlowState.BOOT_SOURCE or inp.global_constant:
                    _add_static_tensor(
                        var_list=var_list,
                        vocab=vocab,
                        level_list=level_list,
                        var_to_id=var_to_id,
                        pairs=pairs,
                        inp=inp)

                var_list.append(vocab[(operation.primitive_name, True, inp.primitive_name)])
                level_list.append(PRIMITIVE_LEVEL)

            for output in operation.outputs:
                var_list.append(vocab[(operation.primitive_name, False, output.primitive_name)])
                level_list.append(PRIMITIVE_LEVEL)
                var_to_id[(operation.name, False, output.name)] = len(var_list) - 1
        elif operation.type == graph_util.OperationType.CONDITIONAL_OPERATION:

            # This counter is representing the sub-graph that it will link to.
            # Unfortunately, each input can mean one of two sub-graphs, so how do we specify both?
            # We do this by saying the inputs go to the If True sub-graph,
            # and the outputs come from the If False sub-graph.

            if_true_level = _convert_graph_to_dataset(dataset, operation, operation.operations_if_true, operation.links_if_true, vocab)

            # The inputs are said to go to the "If True" sub-graph
            for i, inp in enumerate(operation.inputs):
                var_to_id[(operation.name, True, inp.name)] = len(var_list)

                if inp.flow_state == graph_util.FlowState.BOOT_SOURCE or inp.global_constant:
                    _add_static_tensor(
                        var_list=var_list,
                        vocab=vocab,
                        level_list=level_list,
                        var_to_id=var_to_id,
                        pairs=pairs,
                        inp=inp)

                var_list.append(CONDITIONAL_INPUT_ID_OFFSET - i)
                level_list.append(if_true_level)

            if_false_level = _convert_graph_to_dataset(dataset, operation, operation.operations_if_false, operation.links_if_false, vocab)

            # The outputs are said to come from the "negative" graph.
            for i, output in enumerate(operation.outputs):
                var_list.append(CONDITIONAL_OUTPUT_ID_OFFSET - i)
                level_list.append(if_false_level)
                var_to_id[(operation.name, False, output.name)] = len(var_list) - 1

        else:
            # It's composite
            composite_level = _convert_graph_to_dataset(dataset, operation, operation.operations, operation.links, vocab)

            for i, inp in enumerate(operation.inputs):
                var_to_id[(operation.name, True, inp.name)] = len(var_list)

                if inp.flow_state == graph_util.FlowState.BOOT_SOURCE or inp.global_constant:
                    _add_static_tensor(
                        var_list=var_list,
                        vocab=vocab,
                        level_list=level_list,
                        var_to_id=var_to_id,
                        pairs=pairs,
                        inp=inp)

                var_list.append(COMPOSITE_INPUT_ID_OFFSET - i)
                level_list.append(composite_level)

            for i, output in enumerate(operation.outputs):
                var_list.append(COMPOSITE_OUTPUT_ID_OFFSET - i)
                level_list.append(composite_level)
                var_to_id[(operation.name, False, output.name)] = len(var_list) - 1

    for i, output in enumerate(graph.outputs):
        var_list.append(SUB_GRAPH_OUTPUT_ID_OFFSET - i)
        level_list.append(DUMMY_LEVEL)
        var_to_id[("this", True, output.name)] = len(var_list) - 1

    for link in links:
        # Construct adjacency matrix
        if (link.source.operation, False, link.source.data) not in var_to_id:
            print(graph.name)
            raise ValueError(f"Source variable {link.source.operation} {link.source.data} not found. {link.model_dump_json()}")
        source_id = var_to_id[(link.source.operation, False, link.source.data)]
        sink_id = var_to_id[(link.sink.operation, True, link.sink.data)]
        pairs.append((source_id, sink_id))

    check = set(pairs)
    for i in range(len(var_list)):
        row = []
        for j in range(len(var_list)):
            if (i, j) in pairs:
                check.remove((i, j))
                row.append(1)
            else:
                row.append(0)
        adj_matrix.append(row)

    assert check == set(), f"Not all links were used. {graph.name}"

    return this_level



def convert_graph_to_dataset(
        graph: graph_util.Operation,
        vocab: Mapping[Tuple[str, bool, str], int]) -> HierarchicalDataset:

    """ Convert graph to dataset. """

    # This is project level.
    global global_constants
    if graph.global_constants:
        global_constants = {global_constant.name: global_constant for global_constant in graph.global_constants}
    else:
        global_constants = {}

    global counter
    counter = -1

    dataset = HierarchicalDataset(name=graph.name)

    if graph.type == graph_util.OperationType.CONDITIONAL_OPERATION:
        _convert_graph_to_dataset(dataset, graph, graph.operations_if_true, graph.links_if_true, vocab)
        _convert_graph_to_dataset(dataset, graph, graph.operations_if_false, graph.links_if_false, vocab)
    else:
        _convert_graph_to_dataset(dataset, graph, graph.operations, graph.links, vocab)

    return dataset


def create_vocab() -> Mapping[Tuple[str, bool, str], int]:
    """ Create vocabulary. """

    primitives_folder = os.getcwd() + "/../compute_operations/"

    graphs = collect_operations(primitives_folder)
    vocab_map = get_all_vocab(graphs)
    print(f"Vocab length so far: {len(vocab_map)}")
    add_static_tensor_vocab(vocab_map)
    print(f"Vocab length with static tensors: {len(vocab_map)}")
    return vocab_map


def run_all():
    vocab_map = create_vocab()

    # Save vocab to file
    savable_vocab = [(i, key) for key, i in vocab_map.items()]
    add_special_vocab(savable_vocab)

    SAVE_LOCATION = "./graphbook_dataset"

    # os mkdir at save location, overwrite if exists
    # remove save location if exists
    if os.path.exists(SAVE_LOCATION):
        shutil.rmtree(SAVE_LOCATION)

    os.mkdir(SAVE_LOCATION)
    os.mkdir(f"{SAVE_LOCATION}/graphs")

    with open(f"{SAVE_LOCATION}/vocab.json", "w") as f:
        json.dump(savable_vocab, f)

    dataset_folders = [

        os.getcwd() + "/../nlp_models/classifiers/",
        os.getcwd() + "/../nlp_models/generators/",
        os.getcwd() + "/../nlp_models/next_token/",
        os.getcwd() + "/../nlp_models/tokenizers/",
        os.getcwd() + "/../nlp_models/transformers/"
    ]

    composites = [
        os.getcwd() + "/../compute_operations/activation_operations/",
        os.getcwd() + "/../compute_operations/database_operations/",
        os.getcwd() + "/../compute_operations/text_operations/",
        os.getcwd() + "/../compute_operations/transient_operations/",
        os.getcwd() + "/../compute_operations/optimizer_operations/",
        os.getcwd() + "/../compute_operations/math_operations/",
        os.getcwd() + "/../compute_operations/common_layer_operations/",
        os.getcwd() + "/../compute_operations/data_type_parsing_operations/",
        os.getcwd() + "/../compute_operations/loss_function_operations/",
        os.getcwd() + "/../compute_operations/initializer_operations/",
        os.getcwd() + "/../compute_operations/logical_operations/",
        os.getcwd() + "/../compute_operations/shaping_operations/"
    ]

    """
        Each composite graph is a sequence of vocab Ids  and adj matrix

        vocab_ids = [... , ... , ...], -1 for a "composite" vocab.
        adj_matrix = [[... , ... , ...], [... , ... , ...], [... , ... , ...]]
        graph_level_ids = [... , ... , ...]
    """

    for folder in composites + dataset_folders:
        for file in os.listdir(folder):
            if file.endswith(".json"):
                with open(os.path.join(folder, file), "r") as f:
                    graph_json = json.load(f)
                    graph_obj = graph_util.Operation.model_validate(graph_json)
                    logging.info(f"Num Graph Levels:"
                                 f"\t{graph_obj.name}"
                                 f"\t{graph_util.calculate_num_graph_levels(graph_obj)}")

                    logging.info(f"Max Graph Height:"
                                    f"\t{graph_obj.name}"
                                    f"\t{graph_util.calculate_graph_maximum_height(graph_obj)}\n")

                    if graph_obj.type != graph_util.OperationType.PRIMITIVE_OPERATION:
                        dataset = convert_graph_to_dataset(graph_obj, vocab_map)
                        name = graph_obj.name.replace(" ", "_").replace("/", "_")
                        with open(f"{SAVE_LOCATION}/graphs/{name}.json", "w") as f_file:
                            f_file.write(dataset.model_dump_json(exclude_none=True))


def convert_one(full_path: str) -> Tuple[HierarchicalDataset, Mapping[Tuple[str, bool, str], int]]:
    """ Convert one full path graph to dataset."""
    vocab_map = create_vocab()

    with open(full_path, "r") as f:
        graph_json = json.load(f)

    graph_obj = graph_util.Operation.model_validate(graph_json)
    num_levels = graph_util.calculate_num_graph_levels(graph_obj)
    print(num_levels)
    if graph_obj.type != graph_util.OperationType.PRIMITIVE_OPERATION:
        return convert_graph_to_dataset(graph_obj, vocab_map), vocab_map

    return HierarchicalDataset(name=graph_obj.name), vocab_map


if __name__ == "__main__":

    # Argparse has two options, either input a single full file path, or run all and save.
    parser = argparse.ArgumentParser(description='Construct dataset from Graphbook graph.')

    parser.add_argument('--full_path', type=str, help='[Optional] Full path to Graphbook graph.')
    parser.add_argument('--log', type=str, default='INFO', help='Logging level.')

    args = parser.parse_args()

    # set logging level
    logging.basicConfig(level=args.log)

    # If running from script, just uncomment and specify graph.
    # args.full_path = os.getcwd() + "/../nlp_models/classifiers/Fine Tune BERT with Text Classifier.json"

    if args.full_path:
        convert_one(args.full_path)
    else:
        run_all()
