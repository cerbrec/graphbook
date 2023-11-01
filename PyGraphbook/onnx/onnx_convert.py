import pydantic

import os
from typing import List, Dict, Optional

import onnx
import json
from google.protobuf.json_format import MessageToJson
from contextlib import suppress


def prune_long_raw_data_list_values(json_list):
    """Recursively handle lists."""

    for item in json_list:
        if isinstance(item, dict):
            prune_long_raw_data_values(item)
        elif isinstance(item, list):
            prune_long_raw_data_list_values(item)
        else:
            continue

def prune_long_raw_data_values(json_dict):
    """ Recursively go through json key value pairs.

        If encounter rawData key, then check value length. if too long, remove.
    """

    with suppress(KeyError):
        del json_dict["rawData"]

    if isinstance(json_dict, dict):
        for key, value in json_dict.items():
            if isinstance(value, dict):
                prune_long_raw_data_values(value)
            elif isinstance(value, list):
                prune_long_raw_data_list_values(value)
            else:
                continue

    return json_dict


def convert_onnx_to_dict(onnx_file_name: str) -> Dict:
    """ Convert onnx file to dict."""

    onnx_model = onnx.load(onnx_file_name)

    onnx.checker.check_model(onnx_model)

    json_str_model = MessageToJson(onnx_model)

    json_model = json.loads(json_str_model)
    json_model = prune_long_raw_data_values(json_model)

    return json_model


class OnnxTensor(pydantic.BaseModel):
    # Dims is a list of String (int)
    dims: List[str] = pydantic.Field(None, alias="dims")
    dataType: int = pydantic.Field(None, alias="dataType")


class OnnxAttribute(pydantic.BaseModel):
    name: str = pydantic.Field(..., alias="name")
    type: str = pydantic.Field(..., alias="type")
    i: str = pydantic.Field(None, alias="i")
    t: OnnxTensor = pydantic.Field(None, alias="t")
    ints: List[str] = pydantic.Field(None, alias="ints")


class OnnxOperation(pydantic.BaseModel):
    name: str = pydantic.Field(..., alias="name")
    opType: str = pydantic.Field(..., validation_aliases=pydantic.AliasChoices("opType", "type"))

    input: List[str] = pydantic.Field(None, alias="input")
    output: List[str] = pydantic.Field(None, alias="output")

    # Attribute is a list of OnnxAttribute
    attribute: List[OnnxAttribute] = pydantic.Field(None, type=list)


def onnx_to_onnx_list(onnx_file: str) -> List[OnnxOperation]:
    onnx_json = convert_onnx_to_dict(onnx_file)
    return [OnnxOperation(**operation) for operation in onnx_json["graph"]["node"]]


def onnx_folder_to_onnx_list(onnx_folder: str) -> List[OnnxOperation]:
    onnx_list = []
    for file in os.listdir(onnx_folder):
        if file.endswith(".onnx"):
            onnx_list.extend(onnx_to_onnx_list(os.path.join(f"{onnx_folder}/{file}")))
    return onnx_list


if __name__ == "__main__":

    onnx_list = onnx_folder_to_onnx_list("flan-t5-small-onnx")
    print('analyze')

    # names = {op.name.split("/")[-1].split("_")[0] for op in onnx_list}
    op_types = {op.opType for op in onnx_list}
    print(f"Number of op_types: {len(op_types)}")

    # Import counter
    from collections import Counter
    counter = Counter([op.opType for op in onnx_list])

    # Number of inputs on average
    num_inputs = Counter([len(op.input) if op.input else 0 for op in onnx_list])

    from collections import defaultdict
    # Map each operation to set of number of inputs range
    op_to_num_inputs = defaultdict(set)
    op_to_attributes = defaultdict(set)
    for op in onnx_list:
        if op.input:
            op_to_num_inputs[op.opType].add(len(op.input))
        else:
            op_to_num_inputs[op.opType].add(0)

        if op.attribute:
            op_to_attributes[op.opType].add(len(op.attribute))
        else:
            op_to_attributes[op.opType].add(0)

    # Finding: Concat can have 2 or more inputs, Constants have 0

    # Looking at the "If" operation
        # There is an else_branch and a then_branch
        # I think the then_branch is the ifTrue branch
        # Has 33 outputs

    # How many attributes per operation type:
    # print("Attributes per operation type:")
    for op_type in op_types:
        print(f"{op_type}: attributes: {op_to_attributes[op_type]}, num_inputs: {op_to_num_inputs[op_type]}")


    # Determine potential links, how many variables are in the output of one oepration and the input of another?

    links = []
    for op in onnx_list:
        if op.input:
            for input in op.input:
                for oop in onnx_list:
                    if oop.output and op.name != oop.name:
                        for output in oop.output:
                            if input == output:
                                links.append((op, oop, input))

    print(f"Links: {len(links)}")

    print("Check")

    # Number of outputs on average