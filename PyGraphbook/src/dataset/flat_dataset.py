""" Construct flat dataset from Graphbook graph. """

import argparse
import json
import logging
import os
import shutil
from typing import List, Mapping, Tuple, Dict, Optional

from pydantic import BaseModel, Field

from src import graph_util
from src.dataset import variable_vocab

global counter

DATASET_FOLDER = "flat_dataset"
SAVE_LOCATION = f"./{DATASET_FOLDER}"

TOP = "top"
THIS = "this"


def add_special_vocab(vocab: list):
    """ Add special vocab to vocab list. """

    # Add conditional input and loop output types
    vocab.append((len(vocab), ("conditional", True, "if_true")))
    vocab.append((len(vocab), ("loop_body", True, "Run Again")))
    vocab.append((len(vocab), ("loop_body", True, "Looping Data Parent Input")))
    vocab.append((len(vocab), ("loop_body", True, "Looping Data Sub-graph Input")))

    top_index_offset = len(vocab)

    vocab.extend([(top_index_offset + i, (TOP, True, str(i))) for i in range(10)])

    top_index_offset = len(vocab)

    vocab.extend([(top_index_offset + i, (TOP, False, str(i))) for i in range(10)])


def _static_to_var(
        inp: graph_util.Variable,
        vocab: Mapping[Tuple[str, bool, str], int],
) -> int:
    """ Fetch static tensor. """

    shape_length = 0
    if shape_length:
        shape_length = len(inp.shape)
    type_shape_tuple = (str(inp.type), False, str(shape_length))
    if type_shape_tuple not in vocab:
        return -1
    return vocab[type_shape_tuple]


class Dataset(BaseModel):
    """ Graph transformer input.
        vocab_ids = [... , ... , ...],
        adj_matrix = [[... , ... , ...], [... , ... , ...], [... , ... , ...]]
        graph_level_ids = [... , ... , ...]
    """

    name: str = Field(..., description="Name of the Graph.")
    variables: List[int] = Field(default=list(), description="Vocab Ids for Variables")
    adj_pairs: List[Tuple[int, int]] = Field(default=list(), description="Adjacency Pairs")
    adj_matrix: List[List[int]] = Field(default=list(), description="Adjacency Matrix")
    graph_height: List[int] = Field(default=list(), description="Graph height Ids")
    graph_level_ids: List[int] = Field(default=list(), description="Graph level Ids")


def _convert_graph(graph: graph_util.Operation, operations: List[graph_util.Operation], links: List[graph_util.Link],
                   dataset: Dataset,
                   current_level: int, vocab: Mapping[Tuple[str, bool, str], int], input_to_index: Dict[str, int],
                   input_index_to_positional_index: Dict[int, List[int]]) -> Dict[Tuple[str, str], int | List[int]]:
    """ Flatten graph and populate dataset

    Args:
        graph: Graph to flatten's name.
        dataset: Dataset to populate.
        current_level: Current level of graph.
        vocab: Vocab map.
        input_to_index: Map of input name to index.
        input_index_to_positional_index: Map of input index to positional index.

    For each operation, for each input, if in link...
        if source of link is "this" operation, then get the index using input_to_index
            and use supplier_to_index get the original index to make a pair.
        if sink of link is composite/conditional, then recursively convert graph where the supplier to index
            contains the original index, either from within the graph or recursively from the supplier_to_index

        each input and output that is primitive is added to the dataset.
    """

    global counter
    counter += 1
    this_level = int(counter)

    var_to_links = {
        (link.source.operation, link.source.data): (link.sink.operation, link.sink.data) for link in links
    }
    var_to_links.update({
        (link.sink.operation, link.sink.data): (link.source.operation, link.source.data) for link in links
    })

    output_links = {(link.source.operation, link.source.data): (link.sink.operation, link.sink.data)
                    for link in links if link.sink.operation == THIS}

    composite_output_to_index = {}
    primitive_outputs = {}

    for operation in operations:

        if operation.type == graph_util.OperationType.PRIMITIVE_OPERATION:

            for inp in operation.inputs:
                # then this is a new pair.
                var_id = vocab[(operation.primitive_name, True, inp.primitive_name)]
                var_index = len(dataset.variables)

                # Add to the dataset.
                dataset.variables.append(var_id)
                dataset.graph_height.append(current_level)
                dataset.graph_level_ids.append(this_level)

                if (operation.name, inp.name) in var_to_links:
                    # Then there's a link here.
                    link_source = var_to_links[(operation.name, inp.name)]

                    if link_source[0] == THIS:
                        # Then this comes from supplier
                        input_index = input_to_index[link_source[1]]
                        if input_index not in input_index_to_positional_index:
                            # Then it's bootstrapped, so there's nothing that it's coming from.
                            # Here's where we should just use a bootstrapped value.
                            var_id = _static_to_var(inp, vocab)
                            new_source_index = len(dataset.variables)
                            dataset.variables.append(var_id)
                            dataset.graph_height.append(current_level)
                            dataset.graph_level_ids.append(this_level)
                            original_source_indices = [new_source_index]
                        else:
                            original_source_indices = input_index_to_positional_index[input_index]
                        for original_source_index in original_source_indices:
                            dataset.adj_pairs.append((original_source_index, var_index))
                    elif link_source in composite_output_to_index:
                        # Then it's coming from a composite in the same graph.
                        original_source_indices = composite_output_to_index[link_source]
                        for original_source_index in original_source_indices:
                            dataset.adj_pairs.append((original_source_index, var_index))
                    else:
                        # Then it's coming from a primitive.
                        source_id = primitive_outputs[(link_source[0], link_source[1])]
                        dataset.adj_pairs.append((source_id, var_index))

            for out in operation.outputs:
                var_id = vocab[(operation.primitive_name, False, out.primitive_name)]
                var_index = len(dataset.variables)

                # Add to the dataset.
                dataset.variables.append(var_id)
                dataset.graph_height.append(current_level)
                dataset.graph_level_ids.append(this_level)

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

                # If this is conditional input, then we need to add it to the vocab.
                if operation.type == graph_util.OperationType.CONDITIONAL_OPERATION and inp.primitive_name == "if_true" and i == 0:

                    var_index = len(dataset.variables)
                    dataset.variables.append(vocab[("conditional", True, "if_true")])

                    dataset.graph_height.append(current_level)
                    dataset.graph_level_ids.append(this_level)

                    if (operation.name, inp.name) in var_to_links:
                        # Then there's a link here.
                        link_source = var_to_links[(operation.name, inp.name)]

                        if link_source[0] == THIS:
                            # Then this comes from supplier
                            input_index = input_to_index[link_source[1]]
                            if input_index not in input_index_to_positional_index:
                                # Then it's bootstrapped, so there's nothing that it's coming from.
                                # Here's where we should just use a bootstrapped value.

                                var_id = _static_to_var(inp, vocab)
                                new_source_index = len(dataset.variables)
                                dataset.variables.append(var_id)
                                dataset.graph_height.append(current_level)
                                dataset.graph_level_ids.append(this_level)
                                original_source_indices = [new_source_index]
                            else:
                                original_source_indices = input_index_to_positional_index[input_index]
                            for original_source_index in original_source_indices:
                                dataset.adj_pairs.append((original_source_index, var_index))
                        elif link_source in composite_output_to_index:
                            # Then it's coming from a composite in the same graph.
                            original_source_indices = composite_output_to_index[link_source]
                            for original_source_index in original_source_indices:
                                dataset.adj_pairs.append((original_source_index, var_index))
                        else:
                            # Then it's coming from a primitive.
                            source_id = primitive_outputs[(link_source[0], link_source[1])]
                            dataset.adj_pairs.append((source_id, var_index))
                    continue

                # input to index.
                comp_inp_name_to_index[inp.name] = i

                if (operation.name, inp.name) in var_to_links:
                    # Then it's in SOME link.
                    link_source = var_to_links[(operation.name, inp.name)]
                    # This is what's passed into new composite sub-graphs.
                    if link_source[0] == THIS:
                        # Then this comes from supplier
                        input_index = input_to_index[link_source[1]]
                        if input_index not in input_index_to_positional_index:
                            # Then it's bootstrapped, so there's nothing that it's coming from.
                            # Here's where we should just use a bootstrapped value.
                            var_id = _static_to_var(inp, vocab)
                            new_source_index = len(dataset.variables)
                            dataset.variables.append(var_id)
                            dataset.graph_height.append(current_level)
                            dataset.graph_level_ids.append(this_level)
                            original_source_indices = [new_source_index]
                        else:
                            original_source_indices = input_index_to_positional_index[input_index]
                        comp_index_to_index[i] = original_source_indices
                    elif link_source in composite_output_to_index:
                        # Then it's coming from a composite in the same graph.
                        original_source_indices = composite_output_to_index[link_source]
                        comp_index_to_index[i] = original_source_indices
                    elif operation.type == graph_util.OperationType.LOOP_BODY_OPERATION and \
                            link_source[1] in operation.repeat_until_false_condition.loop_data:
                        # this is looping data and doesn't come from anywhere
                        #vocab.append((len(vocab), ("loop_body", True, "Looping Data Parent Input")))
                        var_id = vocab[("loop_body", True, "Looping Data Parent Input")]
                        new_source_index = len(dataset.variables)
                        dataset.variables.append(var_id)
                        dataset.graph_height.append(current_level)
                        dataset.graph_level_ids.append(this_level)
                        # There is no incoming link.
                        comp_index_to_index[i] = [new_source_index]

                    else:
                        # Then it's coming from a primitive.
                        original_source_index = primitive_outputs[(link_source[0], link_source[1])]
                        comp_index_to_index[i] = [original_source_index]
                else:
                    # nothing to do if not supplied,
                    # TODO: add indication of tensor shape/type/value
                    continue

            if operation.type == graph_util.OperationType.CONDITIONAL_OPERATION:
                output_op_and_var_to_index_if_true = _convert_graph(graph=operation,
                                                                    operations=operation.operations_if_true,
                                                                    links=operation.links_if_true, dataset=dataset,
                                                                    current_level=current_level + 1, vocab=vocab,
                                                                    input_to_index=comp_inp_name_to_index,
                                                                    input_index_to_positional_index=comp_index_to_index)

                output_op_and_var_to_index_if_false = _convert_graph(graph=operation,
                                                                     operations=operation.operations_if_false,
                                                                     links=operation.links_if_false, dataset=dataset,
                                                                     current_level=current_level + 1, vocab=vocab,
                                                                     input_to_index=comp_inp_name_to_index,
                                                                     input_index_to_positional_index=comp_index_to_index)

                for key, value in output_op_and_var_to_index_if_false.items():
                    if key in output_op_and_var_to_index_if_true:
                        output_op_and_var_to_index_if_true[key].extend(value)
                    else:
                        output_op_and_var_to_index_if_true[key] = value

                # The same inputs can be supplied by multiple inputs, from either sub-graph.

                composite_output_to_index.update(output_op_and_var_to_index_if_true)

            else:

                # register op
                output_op_and_var_to_index = _convert_graph(graph=operation, operations=operation.operations,
                                                            links=operation.links, dataset=dataset,
                                                            current_level=current_level + 1, vocab=vocab,
                                                            input_to_index=comp_inp_name_to_index,
                                                            input_index_to_positional_index=comp_index_to_index)

                composite_output_to_index.update(output_op_and_var_to_index)

    # Finally, we can get the final outputs
    returnable = {}
    for (link_source_name, link_source_var), (_, link_sink_var) in output_links.items():

        if graph.type == graph_util.OperationType.LOOP_BODY_OPERATION \
                and graph.repeat_until_false_condition and graph.repeat_until_false_condition.name == link_sink_var:

            # Needs to be added as vocab
            var_index = len(dataset.variables)
            dataset.variables.append(vocab[("loop_body", True, "Run Again")])
            dataset.graph_level_ids.append(this_level)
            dataset.graph_height.append(current_level)
            # check input

            if (link_source_name, link_source_var) in composite_output_to_index:
                # Then it's coming from a composite in the same graph.
                original_source_indices = composite_output_to_index[(link_source_name, link_source_var)]
                for original_source_index in original_source_indices:
                    dataset.adj_pairs.append((original_source_index, var_index))
            else:
                # Then it's coming from a primitive.
                source_id = primitive_outputs[(link_source_name, link_source_var)]
                dataset.adj_pairs.append((source_id, var_index))
            continue

        # Determine true source
        if (link_source_name, link_source_var) in composite_output_to_index:
            # Then it's coming from a composite in the same graph.
            original_source_index = composite_output_to_index[(link_source_name, link_source_var)]
        else:
            # Then it's coming from a primitive.
            original_source_index = [primitive_outputs[(link_source_name, link_source_var)]]
        returnable[(graph.name, link_sink_var)] = original_source_index

    """
        vocab.append((len(vocab), ("loop_body", True, "Looping Data Sub-graph Input")))
    """
    if graph_util.OperationType.LOOP_BODY_OPERATION == graph.type:
        for out in graph.outputs:
            if out.name in graph.repeat_until_false_condition.loop_data:
                # Then we need to establish this as well and establish links.
                var_index = len(dataset.variables)
                dataset.variables.append(vocab[("loop_body", True, "Looping Data Sub-graph Input")])
                dataset.graph_level_ids.append(this_level)
                dataset.graph_height.append(current_level)

                if (THIS, out.name) in var_to_links:
                    link_source = var_to_links[(THIS, out.name)]
                    if link_source in composite_output_to_index:
                        # Then it's coming from a composite in the same graph.
                        original_source_indices = composite_output_to_index[link_source]
                        for original_source_index in original_source_indices:
                            dataset.adj_pairs.append((original_source_index, var_index))
                    else:
                        # Then it's coming from a primitive.
                        source_id = primitive_outputs[(link_source[0], link_source[1])]
                        dataset.adj_pairs.append((source_id, var_index))

    return returnable


def convert_graph_to_dataset(
        graph: graph_util.Operation,
        vocab: Mapping[Tuple[str, bool, str], int]) -> Dataset:
    """ Convert graph to dataset. """

    dataset = Dataset(name=graph.name, type=graph.type.name)

    global counter
    counter = -1
    this_level = int(counter)

    supplier_to_index = {}
    input_to_index = {}

    """
    input_to_index: Dict[str, int],
    supplier_to_index: Dict[int, int]) -> Dict[Tuple[str, str], int]:
    """

    # Initially, the inputs and outputs of the operation are the initial and final variables.
    # There are special keywords for these in the vocab
    for i, inp in enumerate(graph.inputs):
        input_to_index[inp.name] = i
        supplier_to_index[i] = [len(dataset.variables)]

        dataset.variables.append(vocab[(TOP, True, str(i))])
        dataset.graph_height.append(0)
        dataset.graph_level_ids.append(this_level)

    if graph.type == graph_util.OperationType.CONDITIONAL_OPERATION:

        # _convert_graph_to_dataset(dataset, graph, graph.operations_if_true, graph.links_if_true, vocab)
        # if_f_level = _convert_graph_to_dataset(dataset, graph, graph.operations_if_false, graph.links_if_false, vocab)
        # dataset.if_false_subgraph_level = if_f_level
        if_true_suppliers = _convert_graph(graph=graph, operations=graph.operations_if_true, links=graph.links_if_true,
                                           dataset=dataset, current_level=0, vocab=vocab, input_to_index=input_to_index,
                                           input_index_to_positional_index=supplier_to_index)

        if_false_suppliers = _convert_graph(graph=graph, operations=graph.operations_if_false,
                                            links=graph.links_if_false, dataset=dataset, current_level=0, vocab=vocab,
                                            input_to_index=input_to_index,
                                            input_index_to_positional_index=supplier_to_index)

        for key, value in if_false_suppliers:
            if key in if_true_suppliers:
                if type(value) is List:
                    if_true_suppliers[key].extend(value)
                else:
                    if_true_suppliers[key] = [if_true_suppliers[key], value]
            else:
                if type(value) is List:
                    if_true_suppliers[key] = value
                else:
                    if_true_suppliers[key] = [value]

        output_op_names_to_suppliers = if_true_suppliers


    else:
        output_op_names_to_suppliers = _convert_graph(graph=graph, operations=graph.operations, links=graph.links,
                                                      dataset=dataset, current_level=0, vocab=vocab,
                                                      input_to_index=input_to_index,
                                                      input_index_to_positional_index=supplier_to_index)

    # Finally, we can get the final outputs
    for i, out in enumerate(graph.outputs):
        dataset.variables.append(vocab[(TOP, False, str(i))])
        dataset.graph_height.append(0)
        dataset.graph_level_ids.append(this_level)

        if (graph.name, out.name) in output_op_names_to_suppliers:
            # Then it's supplied.
            original_source_indices = output_op_names_to_suppliers[(graph.name, out.name)]
            for original_source_index in original_source_indices:
                dataset.adj_pairs.append((original_source_index, len(dataset.variables) - 1))

    print(dataset)

    print(f"Converting {len(dataset.adj_pairs)} adj pairs to matrix...")

    dataset.adj_matrix = [[0 for _ in range(len(dataset.variables))] for _ in range(len(dataset.variables))]

    for pair in dataset.adj_pairs:
        dataset.adj_matrix[pair[0]][pair[1]] = 1

    dataset.adj_pairs = None
    return dataset


def run_all():
    vocab_map = variable_vocab.create_vocab()

    # Save vocab to file
    savable_vocab = [(i, key) for key, i in vocab_map.items()]
    add_special_vocab(savable_vocab)
    vocab_map = {vocab_level[1]: vocab_level[0] for vocab_level in savable_vocab}

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
                        print(graph_obj.name)
                        dataset = convert_graph_to_dataset(graph_obj, vocab_map)
                        name = graph_obj.name.replace(" ", "_").replace("/", "_")
                        with open(f"{SAVE_LOCATION}/graphs/{name}.json", "w") as f_file:
                            f_file.write(dataset.model_dump_json(exclude_none=True))


def convert_one(full_path: str):
    vocab_map = variable_vocab.create_vocab()

    # Save vocab to file
    savable_vocab = [(i, key) for key, i in vocab_map.items()]
    add_special_vocab(savable_vocab)
    vocab_map = {vocab_level[1]: vocab_level[0] for vocab_level in savable_vocab}

    with open(full_path, "r") as f:
        graph_json = json.load(f)
        graph_obj = graph_util.Operation.model_validate(graph_json)
    dataset = convert_graph_to_dataset(graph_obj, vocab_map)
    return dataset, savable_vocab


if __name__ == "__main__":

    # Argparse has two options, either input a single full file path, or run all and save.
    parser = argparse.ArgumentParser(description='Construct dataset from Graphbook graph.')

    parser.add_argument('--full_path', type=str, help='[Optional] Full path to Graphbook graph.')
    parser.add_argument('--log', type=str, default='INFO', help='Logging level.')

    args = parser.parse_args()

    # set logging level
    logging.basicConfig(level=args.log)

    # If running from script, just uncomment and specify graph.
    # args.full_path = os.getcwd() + "/../compute_operations/common_layer_operations/Softmax.json"
    # args.full_path = os.getcwd() + "/../compute_operations/optimizer_operations/Adam Optimizer.json"
    # args.full_path = os.getcwd() + "/../nlp_models/classifiers/Fine Tune BERT with Text Classifier.json"
    # args.full_path = os.getcwd() + "/../nlp_models/generators/GPT-2 Text Generator.json"

    if args.full_path:
        convert_one(args.full_path)
    else:
        run_all()
