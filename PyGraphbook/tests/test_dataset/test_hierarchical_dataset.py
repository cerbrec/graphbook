import json
import os

import pytest

from src import graph_util
from src.dataset import hierarchical_dataset as dataset_util
from src.dataset import reconstruct_hierarchical_graph as reconstruct_util
from src.dataset import variable_vocab as vocab_util


@pytest.mark.parametrize(
    ("dataset_file", "num_levels", "max_height", "num_ops_first_level"),
    [
        pytest.param(
            os.getcwd() + f"/{dataset_util.GRAPH_DATASET_FOLDER_NAME}/graphs/Softmax.json",
            1,
            1,
            9,
        ),
    ],
)
def test_dataset(dataset_file: str, num_levels: int, max_height: int, num_ops_first_level: int):
    """ Test primitive deserialization. """

    with open(os.getcwd() + f"/{dataset_util.GRAPH_DATASET_FOLDER_NAME}/vocab.json", "r") as f:
        vocab = json.load(f)

    # convert vocab which is a nested list into a dict
    vocab = {vocab_level[0]: vocab_level[1] for vocab_level in vocab}

    with open(dataset_file, "r") as f:
        dataset_json = json.load(f)
        dataset_obj = dataset_util.HierarchicalDataset.model_validate(dataset_json)

    # Now let's take dataset_obj and use the vocab to turn it into a graph_util.Operation object.
    op = reconstruct_util.deconstruct_dataset(dataset_obj, vocab)

    assert graph_util.calculate_num_graph_levels(op) == num_levels
    assert graph_util.calculate_graph_maximum_height(op) == max_height
    assert len(op.operations) == num_ops_first_level


@pytest.mark.parametrize(
    "graph_file",
    [
        pytest.param(
            os.getcwd() + "/../compute_operations/common_layer_operations/Softmax.json",
        ),
        pytest.param(
            os.getcwd() + "/../compute_operations/optimizer_operations/Adam Optimizer.json",
        ),
        pytest.param(
            os.getcwd() + "/../nlp_models/transformers/GPT-2.json",
        ),
        pytest.param(
            os.getcwd() + "/../nlp_models/transformers/BERT Uncased.json",
        ),
        pytest.param(
            os.getcwd() + "/../nlp_models/classifiers/Fine Tune BERT with Text Classifier.json",
        ),

    ],
)
def test_construct_and_desconstruct(graph_file: str):
    """ Test primitive deserialization. """

    with open(graph_file, "r") as f:
        graph_json = json.load(f)

    vocab_map = vocab_util.create_vocab()
    savable_vocab = [(i, key) for key, i in vocab_map.items()]
    dataset_util.add_special_vocab(savable_vocab)

    graph_obj = graph_util.Operation.model_validate(graph_json)

    dataset = dataset_util.convert_graph_to_dataset(graph_obj, vocab_map)

    vocab_map = {vocab_level[0]: vocab_level[1] for vocab_level in savable_vocab}
    graph_again = reconstruct_util.deconstruct_dataset(dataset, vocab_map)

    # Same number of levels and operations.
    assert graph_util.calculate_num_graph_levels(graph_obj) == graph_util.calculate_num_graph_levels(graph_again)
    assert graph_util.calculate_graph_maximum_height(graph_obj) == graph_util.calculate_graph_maximum_height(graph_again)
    assert len(graph_obj.operations) == len(graph_again.operations)
    assert graph_util.calculate_num_primitives_in_graph(graph_obj) == graph_util.calculate_num_primitives_in_graph(graph_again)

    # Same number of links.
    assert len(graph_obj.links) == len(graph_again.links)
    assert graph_util.calculate_num_links_in_graph(graph_obj) == graph_util.calculate_num_links_in_graph(graph_again)

