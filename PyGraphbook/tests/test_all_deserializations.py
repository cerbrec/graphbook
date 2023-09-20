import pytest
import json
import os

from src import graph

@pytest.mark.parametrize(
    "folder",
    [
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
        os.getcwd() + "/../compute_operations/shaping_operations/",
    ],
)
def test_all_deserializations(folder: str):
    """ Test all deserializations. """

    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), "r") as f:
                graph_json = json.load(f)
                graph.Operation.model_validate(graph_json)