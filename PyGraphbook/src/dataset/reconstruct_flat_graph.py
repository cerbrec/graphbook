""" Take a (flat) dataset for a graph, and a vocab set, and deconstruct it back into graph form. """

from typing import List, Optional, Tuple
import json

import numpy as np
from src import graph_util
from src.dataset import flat_dataset


DATA_TYPE_KEYWORD = "DataType"
THIS = "this"


def _handle_last_tensor_bootstrapped(variable: graph_util.Variable, last_tensor: tuple):
    """ Handle bootstrapped data. """
    variable.type = last_tensor[0]

    num_dims = int(last_tensor[2])
    # Create nested list which is nested "num_dims" times. Only will be 0 (scalar), 1, or 2.
    if num_dims == 0:
        variable.shape = []
    elif num_dims == 1:
        variable.shape = [None]
    else:
        variable.shape = [[None]]


def _determine_op(
        parent_op: graph_util.Operation,
        operation_name: str,
        is_input: bool,
        var_name: str,
        height: int,
        level: int,
        last_operation_name: str,
        last_was_input: bool,
        last_height: int,
        last_level: int) -> Tuple[graph_util.Operation, graph_util.Operation]:
    """ Determine operation. """

    if operation_name == flat_dataset.TOP:
        return parent_op, parent_op

    if operation_name == last_operation_name and (is_input == last_was_input or last_was_input):
        # Then we're on same operation
        return parent_op.operations[-1], parent_op

    """ 
    vocab.append((len(vocab), (SPECIAL_OPERATION_NAMES[0], True, "if_true")))
    vocab.append((len(vocab), (SPECIAL_OPERATION_NAMES[1], True, "Run Again")))
    vocab.append((len(vocab), (SPECIAL_OPERATION_NAMES[1], True, "Looping Data Parent Input")))
    vocab.append((len(vocab), (SPECIAL_OPERATION_NAMES[1], True, "Looping Data Sub-graph Input")))
    """
    if operation_name == flat_dataset.SPECIAL_OPERATION_NAMES[0]:
        # Then this is a conditional operation
        # Create new operation
        op = graph_util.Operation(
            name=operation_name,
            type=graph_util.OperationType.CONDITIONAL_OPERATION)

    elif operation_name == flat_dataset.SPECIAL_OPERATION_NAMES[1]:
        # Then this is a looping body operation

        # Create new operation
        op = graph_util.Operation(
            name=operation_name,
            type=graph_util.OperationType.LOOP_BODY_OPERATION)

    else:
        # Create new operation
        op = graph_util.Operation(
            name=operation_name,
            type=graph_util.OperationType.PRIMITIVE_OPERATION)

    # if it's on same level as before, then its same parent
    if level == last_level:
        # Then we're on same graph level
        parent_op.operations.append(op)
        return op, parent_op

    # Otherwise, we're on a new graph level
    if last_height == height:
        # Then even though it's not a new height, we're in a new composite.
        # The question is, how many levels did we got up and then down?
        # If we went up and then down, then we need to create a new composite operation
        pass

    if last_height < height:
        # Then we've jumped down
        composite_op = graph_util.Operation(
            name=THIS,
            type=graph_util.OperationType.COMPOSITE_OPERATION)

        parent_op.operations.append(composite_op)

        for i in range(0, height-last_height-1):
            # Create new composite operation
            composite_op_ = graph_util.Operation(
                name=THIS,
                type=graph_util.OperationType.COMPOSITE_OPERATION)
            composite_op.operations.append(composite_op_)
            composite_op = composite_op_

        composite_op.operations.append(op)

        return op, composite_op

    if last_height > height:


    if height == last_height + 1:
        pass
        # Then we're on a new graph level
         = graph_util.Operation(
            name=operation_name,
            type=graph_util.OperationType.COMPOSITE_OPERATION)

        parent_op.operations.append(op)
        return op, op


    if level == last_level:
        # Then we're on same graph level





    # Recursively create operations based on number of levels
    pass

def deconstruct_dataset(
    dataset: flat_dataset.Dataset,
    vocab_: dict
) -> graph_util.Operation:
    """ Deconstruct a flat dataset into a graph.
    """

    # Initially this is composite operation, but if
    top_op = graph_util.Operation(
        name=dataset.name,
        type=graph_util.OperationType.COMPOSITE_OPERATION
    )

    matrix = np.array(dataset.adj_matrix)

    last_operation_name = ""
    last_was_input = False
    last_level = -1
    last_height = -1
    for i, (var_id, height, level) in enumerate(zip(dataset.variables, dataset.graph_height, dataset.graph_level_ids)):
        """
        For each variable, either the variable is on the same graph level as before or on a new graph level
        
        If it's supplied by a variable on a previous graph level, then we link it from THIS
            as we link them from THIS, we append them to an integer list for the new graph level, for each unique ID
    
        If an input variable has multiple incoming links, then it is being supplied by the output of a Conditional operation
        """

        operation_name, is_input, var_name = vocab_[var_id]

        # Determines last op or adds to list if new op.
        op: graph_util.Operation = _determine_op(
            parent_op=top_op,
            operation_name=operation_name,
            is_input=is_input,
            var_name=var_name,
            height=height,
            level=level,
            last_was_input=last_was_input,
            last_operation_name=last_operation_name,
            last_height=last_height,
            last_level=last_level)

        # Determine if we are at a new graph level
        if level != last_level:


        # If it's an input variable, then we need to add it to the inputs of the operation.
        if is_input:
            op.inputs.append(graph_util.Variable(
                name=var_name,
                primitive_name=var_name
            ))

            link_sources = np.where(matrix[:, i] == 1)[0]

            if len(link_sources) == 0:
                # Then it has no input link.



        else:
            op.outputs.append(graph_util.Variable(
                name=var_name,
                primitive_name=var_name
            ))

        # height determines how many levels we went down
        # level is unique identifier of graph level

        # Should be guaranteed that every var_id is in vocab
        op_name, is_input, var_name = vocab_[var_id]

        if op_name == flat_dataset.TOP:
            if is_input:
                top_op.inputs.append(graph_util.Variable(
                    name=var_name,
                    primitive_name=var_name
                ))

                # each input is a column in the adjacency matrix
                link_sources = np.where(dataset.adj_matrix[:, i] == 1)
                if link_sources.empty():
                    # Then it has no input link.
                    pass


                # Where did it get link from, if anywhere?
                # dataset.adj_matrix[i]
            else:
                top_op.outputs.append(graph_util.Variable(
                    name=var_name,
                    primitive_name=var_name
                ))


        if level == last_level:
            # Then we're on same graph level
            pass


    return top_op

if __name__ == "__main__":
    # get file
    dataset_file = "flat_dataset/graphs/Softmax.json"
    with open(dataset_file, "r") as f:
        dataset_json = json.load(f)
        dataset_obj = flat_dataset.Dataset.model_validate(dataset_json)

    # get vocab
    vocab_file = "flat_dataset/vocab.json"
    with open(vocab_file, "r") as f:
        vocab = json.load(f)
        vocab = {vocab_level[0]: vocab_level[1] for vocab_level in vocab}

    result = deconstruct_dataset(dataset_obj, vocab)