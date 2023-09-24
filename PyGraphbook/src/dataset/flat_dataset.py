""" Construct flat dataset from Graphbook graph. """

import json
import os
import shutil
import argparse
import logging

from pydantic import BaseModel, Field
from typing import List, Mapping, Tuple, Dict, Optional

from src import graph_util
from src.dataset import variable_vocab

global counter
global global_constants


SAVE_LOCATION = "./flat_dataset"



class Dataset(BaseModel):
    """ Graph transformer input.
        vocab_ids = [... , ... , ...],
        adj_matrix = [[... , ... , ...], [... , ... , ...], [... , ... , ...]]
        graph_level_ids = [... , ... , ...]
    """

    name: str = Field(..., description="Name of the Graph.")
    variables: List[int] = Field(default=list(), description="Vocab Ids for Variables")
    adj_matrix: List[List[int]] = Field(default=list(), description="Adjacency Matrix")
    graph_level_ids: List[int] = Field(default=list(), description="Graph Level Ids")


def _convert_graph(
        graph: graph_util.Operation,
        dataset: Dataset,
        current_level: int,
        vocab: dict,
        input_to_index: Dict[str, int],
        supplier_to_index: Dict[int, int]) -> None:
    """ Flatten graph and populate dataset

    For each operation, for each input, if in link...
        if source of link is "this" operation, then get the index using input_to_index
            and use supplier_to_index get the original index to make a pair.
        if sink of link is composite/conditional, then recursively convert graph where the supplier to index
            contains the original index, either from within the graph or recursively from the supplier_to_index

        each input and output that is primitive is added to the dataset.
    """

    var_to_links = {
        (link.source.operation, link.source.data): link.sink for link in graph.links
    }
    var_to_links.update({
        (link.sink.operation, link.sink.data): link.source for link in graph.links
    })

    output_links = {k: v for k, v in var_to_links.items() if v == "this"}

    pairs = []
    composite_output_to_index = {}
    primitive_outputs = {}

    for operation in graph.operations:

        if operation.type == graph_util.OperationType.PRIMITIVE_OPERATION:

            for inp in operation.inputs:
                # then this is a new pair.
                var_id = vocab[(operation.primitive_name, True, inp.primitive_name)]
                var_index = len(dataset.variables)

                # Add to the dataset.
                dataset.variables.append(var_id)
                dataset.graph_level_ids.append(current_level)

                if (operation.name, inp.name) in var_to_links:
                    # Then there's a link here.
                    link_source = var_to_links[(operation.name, inp.name)]

                    if link_source.operation == "this":
                        # Then this comes from supplier
                        original_source_index = supplier_to_index[input_to_index[link_source.data]]
                        pairs.append((original_source_index, var_index))
                    elif link_source in composite_output_to_index:
                        # Then it's coming from a composite in the same graph.
                        original_source_index = composite_output_to_index[link_source]
                        pairs.append((original_source_index, var_index))
                    else:
                        # Then it's coming from a primitive.
                        source_id = primitive_outputs[(link_source.operation, link_source.data)]
                        pairs.append((source_id, var_index))

            for out in operation.outputs:
                var_id = vocab[(operation.primitive_name, True, out.primitive_name)]
                var_index = len(dataset.variables)

                # Add to the dataset.
                dataset.variables.append(var_id)
                dataset.graph_level_ids.append(current_level)

                primitive_outputs[(operation.name, out.name)] = var_index

        else:
            comp_inp_to_index = {}
            for i, inp in enumerate(operation.inputs):
                if (operation.name, inp.name) in var_to_links:
                    comp_inp_to_index[i] = var_to_links[]
                else:



            _convert_graph(
                graph=operation,
                dataset=dataset,
                current_level=current_level + 1,
                vocab=vocab,
                input_to_index={op.inp}
            )




            # else:
            #     composite_output_to_index[(operation.name, out.name)]


def convert_graph_to_dataset(
        graph: graph_util.Operation,
        vocab: Mapping[Tuple[str, bool, str], int]) -> Dataset:

    """ Convert graph to dataset. """

    # This is project level.
    global global_constants
    if graph.global_constants:
        global_constants = {global_constant.name: global_constant for global_constant in graph.global_constants}
    else:
        global_constants = {}

    global counter
    counter = -1

    dataset = Dataset(name=graph.name, type=graph.type.name)

    if graph.type == graph_util.OperationType.CONDITIONAL_OPERATION:
        _convert_graph_to_dataset(dataset, graph, graph.operations_if_true, graph.links_if_true, vocab)
        if_f_level = _convert_graph_to_dataset(dataset, graph, graph.operations_if_false, graph.links_if_false, vocab)
        dataset.if_false_subgraph_level = if_f_level
    else:
        _convert_graph_to_dataset(dataset, graph, graph.operations, graph.links, vocab)

    return dataset





def run_all():
    vocab_map = variable_vocab.create_vocab()

    # Save vocab to file
    savable_vocab = [(i, key) for key, i in vocab_map.items()]

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


def convert_one(full_path: str) -> Tuple[Dataset, Mapping[Tuple[str, bool, str], int]]:
    """ Convert one full path graph to dataset."""
    vocab_map = variable_vocab.create_vocab()

    with open(full_path, "r") as f:
        graph_json = json.load(f)

    graph_obj = graph_util.Operation.model_validate(graph_json)
    num_levels = graph_util.calculate_num_graph_levels(graph_obj)
    print(num_levels)
    if graph_obj.type != graph_util.OperationType.PRIMITIVE_OPERATION:
        return convert_graph_to_dataset(graph_obj, vocab_map), vocab_map

    return Dataset(name=graph_obj.name), vocab_map


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
