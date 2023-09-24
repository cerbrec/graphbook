""" Construct flat dataset from Graphbook graph. """

import argparse
import json
import logging
import os
import shutil
from typing import List, Mapping, Tuple, Dict

from pydantic import BaseModel, Field

from src import graph_util
from src.dataset import variable_vocab

global counter
global global_constants


DATASET_FOLDER = "flat_dataset"
SAVE_LOCATION = f"./{DATASET_FOLDER}"

THIS = "this"


def add_special_vocab(vocab: list):
    """ Add special vocab to vocab list. """

    top_index_offset = len(vocab)

    vocab.extend([(top_index_offset + i, ("top", True, str(i))) for i in range(10)])

    top_index_offset = len(vocab)

    vocab.extend([(top_index_offset + i, ("top", False, str(i))) for i in range(10)])


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
        vocab: Mapping[Tuple[str, bool, str], int],
        input_to_index: Dict[str, int],
        supplier_to_index: Dict[int, int]) -> Dict[Tuple[str, str], int]:
    """ Flatten graph and populate dataset

    Args:
        graph: Graph to flatten.
        dataset: Dataset to populate.
        current_level: Current level of graph.
        vocab: Vocab map.
        input_to_index: Map of input name to index.
        supplier_to_index: Map of supplier index to index.

    For each operation, for each input, if in link...
        if source of link is "this" operation, then get the index using input_to_index
            and use supplier_to_index get the original index to make a pair.
        if sink of link is composite/conditional, then recursively convert graph where the supplier to index
            contains the original index, either from within the graph or recursively from the supplier_to_index

        each input and output that is primitive is added to the dataset.
    """

    var_to_links = {
        (link.source.operation, link.source.data): (link.sink.operation, link.sink.data) for link in graph.links
    }
    var_to_links.update({
        (link.sink.operation, link.sink.data): (link.source.operation, link.source.data) for link in graph.links
    })

    output_links = {k: v for k, v in var_to_links.items() if v == THIS}

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

                    if link_source.operation == THIS:
                        # Then this comes from supplier
                        original_source_index = supplier_to_index[input_to_index[link_source.data]]
                        dataset.adj_matrix.append([original_source_index, var_index])
                    elif link_source in composite_output_to_index:
                        # Then it's coming from a composite in the same graph.
                        original_source_index = composite_output_to_index[link_source]
                        dataset.adj_matrix.append([original_source_index, var_index])
                    else:
                        # Then it's coming from a primitive.
                        source_id = primitive_outputs[(link_source.operation, link_source.data)]
                        dataset.adj_matrix.append([source_id, var_index])

            for out in operation.outputs:
                var_id = vocab[(operation.primitive_name, True, out.primitive_name)]
                var_index = len(dataset.variables)

                # Add to the dataset.
                dataset.variables.append(var_id)
                dataset.graph_level_ids.append(current_level)

                primitive_outputs[(operation.name, out.name)] = var_index

        else:
            comp_inp_name_to_index = {}
            comp_index_to_index = {}
            # For each composite input, we need to send in the original index.
            # Can come from 1 of 3 places (if anywhere:
            #   1. From a primitive in the same graph.
            #   2. From a composite in the same graph.
            #   3. From "this" in parent level.

            for i, inp in enumerate(operation.inputs):

                # input to index.
                comp_inp_name_to_index[inp.name] = i

                if (operation.name, inp.name) in var_to_links:
                    # Then it's in SOME link.
                    link_source = var_to_links[(operation.name, inp.name)]

                    original_source_index = -1
                    if link_source.operation == THIS:
                        # Then this comes from supplier
                        original_source_index = supplier_to_index[input_to_index[link_source.data]]
                    elif link_source in composite_output_to_index:
                        # Then it's coming from a composite in the same graph.
                        original_source_index = composite_output_to_index[link_source]
                    else:
                        # Then it's coming from a primitive.
                        original_source_index = primitive_outputs[(link_source.operation, link_source.data)]

                    comp_index_to_index[i] = original_source_index

                else:
                    continue

            # register op
            output_op_and_var_to_index = _convert_graph(
                graph=operation,
                dataset=dataset,
                current_level=current_level + 1,
                vocab=vocab,
                input_to_index=comp_inp_name_to_index,
                supplier_to_index=comp_index_to_index,
            )

            composite_output_to_index.update(output_op_and_var_to_index)

    # Finally, we can get the final outputs
    returnable = {}
    for (link_source_name, link_source_var), (_, link_sink_var) in output_links.items():
        # Determine true source
        if (link_source_name, link_source_var) in composite_output_to_index:
            # Then it's coming from a composite in the same graph.
            original_source_index = composite_output_to_index[(link_source_name, link_source_var)]
        else:
            # Then it's coming from a primitive.
            original_source_index = primitive_outputs[(link_source_name, link_source_var)]
        returnable[(graph.name, link_sink_var)] = original_source_index

    return returnable


def convert_graph_to_dataset(
        graph: graph_util.Operation,
        vocab: Mapping[Tuple[str, bool, str], int]) -> Dataset:

    """ Convert graph to dataset. """

    dataset = Dataset(name=graph.name, type=graph.type.name)

    supplier_to_index = {}
    input_to_index = {}

    """
    input_to_index: Dict[str, int],
    supplier_to_index: Dict[int, int]) -> Dict[Tuple[str, str], int]:
    """

    # Initially, the inputs and outputs of the operation are the initial and final variables.
    # There are special keywords for these in the vocab
    for i, inp in enumerate(graph.inputs):
        supplier_to_index[inp.name] = len(dataset.variables)
        input_to_index[inp.name] = i

        dataset.variables.append(vocab[("top", True, str(i))])
        dataset.graph_level_ids.append(0)

    if graph.type == graph_util.OperationType.CONDITIONAL_OPERATION:
        output_op_names_to_suppliers = {}
        # _convert_graph_to_dataset(dataset, graph, graph.operations_if_true, graph.links_if_true, vocab)
        # if_f_level = _convert_graph_to_dataset(dataset, graph, graph.operations_if_false, graph.links_if_false, vocab)
        # dataset.if_false_subgraph_level = if_f_level
    else:
        output_op_names_to_suppliers = _convert_graph(
            graph=graph,
            dataset=dataset,
            current_level=0,
            vocab=vocab,
            input_to_index=input_to_index,
            supplier_to_index=supplier_to_index)

    # Finally, we can get the final outputs
    for i, out in enumerate(graph.outputs):
        dataset.variables.append(vocab[("top", False, str(i))])
        dataset.graph_level_ids.append(0)

        if output_op_names_to_suppliers[(graph.name, out.name)] in supplier_to_index:
            # Then it's supplied.
            original_source_index = output_op_names_to_suppliers[(graph.name, out.name)]
            dataset.adj_matrix.append([original_source_index, len(dataset.variables) - 1])




    return dataset


def run_all():
    vocab_map = variable_vocab.create_vocab()

    # Save vocab to file
    savable_vocab = [(i, key) for key, i in vocab_map.items()]
    add_special_vocab(savable_vocab)

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
        pass
    else:
        run_all()
