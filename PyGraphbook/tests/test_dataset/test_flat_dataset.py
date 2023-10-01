import json
import os

import pytest

from src import graph_util
from src.dataset import flat_dataset
from src.dataset import reconstruct_flat_graph as reconstruct_util
from src.dataset import variable_vocab as vocab_util


@pytest.mark.parametrize(
    ("dataset_file", "num_operations", "num_links"),
    [
        pytest.param(
            os.getcwd() + f"/{flat_dataset.GRAPH_DATASET_FOLDER_NAME}/graphs/Softmax.json",
            9,
            17,
        ),
        pytest.param(
            os.getcwd() + f"/{flat_dataset.GRAPH_DATASET_FOLDER_NAME}/graphs/Adam_Optimizer.json",
            49,
            72,
        ),
    ],
)
def test_dataset(dataset_file: str, num_operations: int, num_links: int):
    """ Test primitive deserialization. """

    with open(os.getcwd() + f"/{flat_dataset.GRAPH_DATASET_FOLDER_NAME}/vocab.json", "r") as f:
        vocab = json.load(f)

    # convert vocab which is a nested list into a dict
    vocab = {vocab_level[0]: vocab_level[1] for vocab_level in vocab}

    with open(dataset_file, "r") as f:
        dataset_json = json.load(f)
        dataset_obj = flat_dataset.Dataset.model_validate(dataset_json)

    # Now let's take dataset_obj and use the vocab to turn it into a graph_util.Operation object.
    op = reconstruct_util.reconstruct_dataset(dataset_obj, vocab)

    assert len(op.operations) == num_operations
    assert len(op.links) == num_links


@pytest.mark.parametrize(
    "graph_file",
    [
        pytest.param(
            os.getcwd() + "/../compute_operations/common_layer_operations/Softmax.json",
        ),
        pytest.param(
            os.getcwd() + "/../compute_operations/optimizer_operations/Adam Optimizer.json",
        ),
        # pytest.param(
        #     os.getcwd() + "/../nlp_models/transformers/GPT-2.json",
        # ),
        # pytest.param(
        #     os.getcwd() + "/../nlp_models/transformers/BERT Uncased.json",
        # ),
        # pytest.param(
        #     os.getcwd() + "/../nlp_models/classifiers/Fine Tune BERT with Text Classifier.json",
        # ),

    ],
)
def test_construct_and_desconstruct(graph_file: str):
    """ Test primitive deserialization. """

    with open(graph_file, "r") as f:
        graph_json = json.load(f)

    vocab_map = vocab_util.create_vocab()

    # Save vocab to file
    savable_vocab = [(i, key) for key, i in vocab_map.items()]
    flat_dataset.add_special_vocab(savable_vocab)
    vocab_map = {vocab_level[1]: vocab_level[0] for vocab_level in savable_vocab}

    graph_obj = graph_util.Operation.model_validate(graph_json)

    dataset = flat_dataset.convert_graph_to_dataset(graph_obj, vocab_map)

    vocab_map = {vocab_level[0]: vocab_level[1] for vocab_level in savable_vocab}
    graph_again = reconstruct_util.reconstruct_dataset(dataset, vocab_map)

    # Same number of levels and operations.
    assert len(graph_obj.operations) == len(graph_again.operations)

    # Same number of links.
    assert len(graph_obj.links) == len(graph_again.links)

