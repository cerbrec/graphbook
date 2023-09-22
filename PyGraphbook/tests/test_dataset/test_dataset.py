import pytest
import json
import os

import pytest
import json

from src import graph_util
from src.dataset import construct_dataset as dataset_util
from src.dataset import deconstruct_dataset as deconstruct_util


@pytest.mark.parametrize(
    ("dataset_file", "num_levels", "max_height", "num_ops_first_level"),
    [
        pytest.param(
            os.getcwd() + "/graphbook_dataset/graphs/Softmax.json",
            1,
            1,
            14,
        ),
    ],
)
def test_dataset(dataset_file: str, num_levels: int, max_height: int, num_ops_first_level: int):
    """ Test primitive deserialization. """
    with open(os.getcwd() + "/graphbook_dataset/vocab.json", "r") as f:
        vocab = json.load(f)

    # convert vocab which is a nested list into a dict
    vocab = {vocab_level[0]: vocab_level[1] for vocab_level in vocab}

    with open(os.getcwd() + "/graphbook_dataset/graphs/Softmax.json", "r") as f:
        dataset_json = json.load(f)
        dataset_obj = dataset_util.HierarchicalDataset.model_validate(dataset_json)

    # Now let's take dataset_obj and use the vocab to turn it into a graph_util.Operation object.
    op = deconstruct_util.deconstruct_dataset(dataset_obj, vocab)

    assert graph_util.calculate_num_graph_levels(op) == num_levels
    assert graph_util.calculate_graph_maximum_height(op) == max_height
    assert len(op.operations) == num_ops_first_level


@pytest.mark.parametrize(
    "graph_file",
    [
        pytest.param(
            os.getcwd() + "/../compute_operations/common_layer_operations/Softmax.json",
        ),
    ],
)
def test_construct_and_desconstruct(graph_file: str):
    """ Test primitive deserialization. """

    with open(graph_file, "r") as f:
        graph_json = json.load(f)

    vocab_map = dataset_util.create_vocab()
    savable_vocab = [(i, key) for key, i in vocab_map.items()]
    dataset_util.add_special_vocab(savable_vocab)

    graph_obj = graph_util.Operation.model_validate(graph_json)

    dataset = dataset_util.convert_graph_to_dataset(graph_obj, vocab_map)

    vocab_map = {vocab_level[0]: vocab_level[1] for vocab_level in savable_vocab}
    graph_again = deconstruct_util.deconstruct_dataset(dataset, vocab_map)

    assert graph_util.calculate_num_graph_levels(graph_obj) == graph_util.calculate_num_graph_levels(graph_again)
    assert graph_util.calculate_graph_maximum_height(graph_obj) == graph_util.calculate_graph_maximum_height(graph_again)
    assert len(graph_obj.operations) == len(graph_again.operations)
