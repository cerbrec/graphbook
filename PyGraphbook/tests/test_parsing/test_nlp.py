import pytest
import json
import os

from src import graph_util

@pytest.mark.parametrize(
    "folder",
    [
        os.getcwd() + "/../nlp_models/classifiers/",
        os.getcwd() + "/../nlp_models/classifiers/",
        os.getcwd() + "/../nlp_models/next_token/",
        os.getcwd() + "/../nlp_models/tokenizers/",
        os.getcwd() + "/../nlp_models/transformers/",
    ],
)
def test_nlp_deserializations(folder: str):
    """ Test nlp deserializations. """

    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), "r") as f:
                graph_json = json.load(f)
                graph_util.Operation.model_validate(graph_json)