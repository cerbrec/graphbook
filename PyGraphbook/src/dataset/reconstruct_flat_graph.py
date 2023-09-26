""" Take a (flat) dataset for a graph, and a vocab set, and deconstruct it back into graph form. """

from typing import List, Optional

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


def deconstruct_dataset(
    dataset: flat_dataset.Dataset,
    vocab_: dict
) -> graph_util.Operation:
    """ Deconstruct a flat dataset into a graph.

    For each variable, either the variable is on the same graph level as before or on a new graph level
    If it's supplied by a variable on a previous graph level, then we link it from THIS
        as we link them from THIS, we append them to an integer list for the new graph level, for each unique ID

    If an input variable has multiple incoming links, then it is being supplied by the output of a Conditional operation
    """

    # Initially this is composite operation, but if
    top_op = graph_util.Operation(
        name=dataset.name,
        type=graph_util.OperationType.COMPOSITE_OPERATION
    )

    last_level = -1
    last_height = -1
    for i, (var_id, height, level) in enumerate(zip(dataset.variables, dataset.graph_height, dataset.graph_level_ids)):
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
                # find the
                link_sources = np.where(dataset.adj_matrix[:, i] == 1)

                # Where did it get link from, if anywhere?
                # dataset.adj_matrix[i]
            else:
                top_op.outputs.append(graph_util.Variable(
                    name=var_name,
                    primitive_name=var_name
                ))


        if level == last_level:
            # Then we're on same graph level


    return top_op
