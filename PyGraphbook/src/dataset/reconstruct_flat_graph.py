""" Take a (flat) dataset for a graph, and a vocab set, and deconstruct it back into graph form. """

from typing import List, Optional

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

    last_level = -1
    for var in zip(dataset.variables, dataset.graph_level_ids):
        pass

    pass
    # return graph_util.Operation()

