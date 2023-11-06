
from pydantic import BaseModel, Field, AliasChoices
import os
import onnx
import json
import numpy as np

from onnx import numpy_helper
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from google.protobuf.json_format import MessageToJson
from contextlib import suppress
import enum

from netron.onnx import ModelFactory

#pylint: disable=protected-access
from netron.onnx import _Model as NetronModel

# Import graphbook graph
from src import graph as graphbook

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


class ParsedOnnxFile:
    def __init__(self, file_name: str, json_model: Dict, tensor_map: Dict, netron_model: NetronModel):
        self.file_name = file_name
        self.json_model = json_model
        self.tensor_map = tensor_map
        self.netron_model = netron_model

        # This is needed to compile the netron model as well.
        if netron_model:
            self.netron_json = netron_model.to_json()
        else:
            self.netron_json = {}


def convert_onnx_to_dict(onnx_file_name: str) -> ParsedOnnxFile:
    """ Convert onnx file to dict."""

    onnx_model = onnx.load(onnx_file_name)
    onnx.checker.check_model(onnx_model)

    tensor_map = {}
    for tensor in onnx_model.graph.initializer:
        try:
            tensor_map[tensor.name] = numpy_helper.to_array(tensor)
        except Exception as e:
            print("moving on from tensor: ", tensor.name)
            continue

    json_str_model = MessageToJson(onnx_model)

    json_model = json.loads(json_str_model)
    json_model = prune_long_raw_data_values(json_model)

    netron_model = ModelFactory().open(model=onnx_model)

    return ParsedOnnxFile(
        file_name=onnx_file_name,
        json_model=json_model,
        tensor_map=tensor_map,
        netron_model=netron_model
    )


class OnnxTensor(BaseModel):
    # Dims is a list of String (int)
    dims: List[str] = Field(None, alias="dims")
    dataType: int = Field(None, alias="dataType")


class OnnxAttribute(BaseModel):
    name: str = Field(..., alias="name")
    type: str = Field(..., alias="type")
    i: str = Field(None, alias="i")
    t: OnnxTensor = Field(None, alias="t")
    ints: List[str] = Field(None, alias="ints")


class OnnxOperation(BaseModel):
    name: str = Field(..., alias="name")
    opType: str = Field(..., validation_aliases=AliasChoices("opType", "type"))

    input: List[str] = Field(None, alias="input")
    output: List[str] = Field(None, alias="output")

    bootstrap_map: Optional[Dict[str, OnnxAttribute]] = Field(None, alias="bootstrap_map")

    # Attribute is a list of OnnxAttribute
    attribute: List[OnnxAttribute] = Field(None, type=list)

    tensor_map: Optional[Dict[str, List]] = Field(None, alias="tensor_map")
    composite_path: Optional[str] = Field(None, alias="composite_path")

    op_type_meta_data: Optional[Dict] = Field(None, alias="op_type_meta_data")


def create_read_from_file(file_name: str, tensor: List) -> OnnxOperation:
    """ Creates an OnnxOperation for the read from file operation, to use for Graphbook"""
    split_name = file_name.split(".")
    dir_name = ".".join(split_name[:-1])

    read_from_file = OnnxOperation(**{
        'name': file_name + ".read_from_file",
        'opType': "read_from_file",
        'input': [split_name[-1], dir_name, "{}"],
        'output': [file_name]
    })
    read_from_file.tensor_map = {file_name: tensor}

    return read_from_file


def create_write_to_file(file_name: str) -> OnnxOperation:
    split_name = file_name.split(".")
    dir_name = ".".join(split_name[:-1])
    write_to_file = OnnxOperation(**{
        'name': file_name + ".write_to_file",
        'opType': "write_to_file",
        'input': [split_name[0], dir_name, "true", ''],
        'output': []
    })
    return write_to_file


class OnnxLink:
    def __init__(self, source: str, source_int: int, var_name: str, sink_int: int, sink: str):
        self.source = source
        self.source_int = source_int
        self.var_name = var_name
        self.sink_int = sink_int
        self.sink = sink

    # To string
    def __str__(self):
        return f"{self.source} ({self.source_int}) --{self.var_name}--> ({self.sink_int}) {self.sink}"

    def __repr__(self):
        return str(self)


class OnnxGraph:
    def __init__(
            self,
            name: str,
            inputs: set,
            outputs: set,
            onnx_ops: List[OnnxOperation],
            onnx_links: List[OnnxLink],
            parsed_onnx_file: Optional[ParsedOnnxFile] = None):

        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.onnx_ops = onnx_ops
        self.onnx_links = onnx_links
        self.parsed_onnx_file = parsed_onnx_file

    # To string
    def __str__(self):
        # Number of operations and links
        return f"{self.name}: Number of operations: {len(self.onnx_ops)}, Number of links: {len(self.onnx_links)}"

    def __repr__(self):
        return str(self.name)

# class OnnxGraph:
#     def __init__(self, onnx_ops: List[OnnxOperation], links: List[Tuple[str, str, str]]):
#         pass


def get_args_from_netron_op_inputs(op_name: str, netron_model: NetronModel, netron_dict: Dict) -> List[str]:
    """ Get op inputs from netron model."""
    op_inputs = []

    for inp_dict in netron_dict[op_name]["inputs"]:
        if 'arguments' not in inp_dict:
            raise ValueError("NO ARGUMENTS FIELD IN INPUT " + str(inp_dict))
        inp_num = inp_dict['arguments'][0]
        inp = netron_model.graph.arguments[inp_num]
        op_inputs.append(inp.name)
    return op_inputs


def _fetch_tensor_type_for_onnx_tensor(netron_model: NetronModel, opt_type: str, onnx_tensor: OnnxTensor) -> str:
    """ Fetch the tensor type for the onnx tensor."""

    type_list = list(netron_model.metadata.metadata[opt_type]["type_constraints"][-1]["allowed_type_strs"])
    if not isinstance(onnx_tensor.dataType, int) or not isinstance(type_list, list) or onnx_tensor.dataType-1 >= len(type_list):
        raise ValueError("Invalid data type for onnx tensor: " + str(onnx_tensor))

    return type_list[onnx_tensor.dataType-1]


def onnx_to_graph(onnx_file: str) -> Optional[OnnxGraph]:
    """ Convert given onnx file to list of OnnxOperation objects. and tensormap."""
    try:
        parsed_onnx = convert_onnx_to_dict(onnx_file)
    except Exception as e:
        print(f"Could not convert {onnx_file} to NetronModel.")
        return None
    onnx_json = parsed_onnx.json_model
    tensor_map = parsed_onnx.tensor_map
    netron_json = parsed_onnx.netron_json

    netron_dict = {node["name"]: node for node in netron_json["graphs"][0]["nodes"]}
    # This dict is a map of operation outputs to their host.
    # if a name is not there, then it is an empty "input"
    output_to_op = {
        parsed_onnx.netron_model.graph.arguments[op_output['arguments'][0]].name: op["name"]
        for op in netron_json["graphs"][0]["nodes"]
        for op_output in op["outputs"]
    }

    # onnx_dict = {node["name"]: node for node in onnx_json["graph"]["node"]}
    # First, we get all the operations and convert them to OnnxOperation objects.
    onnx_list = []
    link_list = []
    constants = {}
    unfilled_inputs = set()
    read_cache = set()
    write_cache = set(out for out in output_to_op.keys() if out.startswith("present"))
    op_name_to_op = dict()
    for op in onnx_json["graph"]["node"]:

        onnx_op = OnnxOperation(**op)
        onnx_op.op_type_meta_data = dict(parsed_onnx.netron_model.metadata.metadata[onnx_op.opType])
        if len(onnx_op.name.split("/")) > 1:
            onnx_op.composite_path = "/".join(onnx_op.name.split("/")[:-1])

        if onnx_op.attribute:
            for attribute in onnx_op.attribute:
                if attribute.t and attribute.t.dataType:
                    data_type_str = _fetch_tensor_type_for_onnx_tensor(parsed_onnx.netron_model, onnx_op.opType, attribute.t)
                    attribute.t.dataType = data_type_str

        op_name_to_op[onnx_op.name] = onnx_op
        incoming_inputs = get_args_from_netron_op_inputs(onnx_op.name, parsed_onnx.netron_model, netron_dict)
        include_tensor = True
        for i, inp in enumerate(incoming_inputs):
            if inp not in output_to_op:
                if inp.startswith("past"):
                    reformat_inp = ".".join(inp.split(".")[1:])
                    candidates = {cache_name for cache_name in write_cache if cache_name.endswith(reformat_inp)}
                    if len(candidates) == 1:
                        read_cache.add((inp, candidates.pop()))
                    else:
                        read_cache.add((inp, None))
                elif inp not in tensor_map:
                    unfilled_inputs.add(inp)
                    link = OnnxLink(
                        source=onnx_file,
                        source_int=len(unfilled_inputs)-1,
                        var_name=inp,
                        sink_int=i,
                        sink=onnx_op.name
                    )
                    link_list.append(link)
                else:
                    # This is in tensor map. we should just read from file.
                    read_from_file = create_read_from_file(inp, tensor_map[inp])
                    read_from_file.composite_path = onnx_op.composite_path
                    if onnx_op.composite_path:
                        read_from_file.name = onnx_op.composite_path + "/" + read_from_file.name

                    if len(onnx_op.name.split("/")) > 1:
                        # Get the path
                        composite_path = "/".join(onnx_op.name.split("/")[:-1])
                        read_from_file.composite_path = composite_path

                    onnx_list.append(read_from_file)
                    op_name_to_op[read_from_file.name] = onnx_op
                    # Now add Link
                    link = OnnxLink(
                        source=read_from_file.name,
                        source_int=0,
                        var_name=inp,
                        sink_int=i,
                        sink=onnx_op.name
                    )
                    link_list.append(link)
                    include_tensor = False
            else:
                # Add Link
                previous_op = output_to_op[inp]

                if previous_op not in op_name_to_op:
                    print(f"Could not find previous op: {previous_op}")
                    raise ValueError("Could not find previous op: " + previous_op)
                    # check_back_later.append((onnx_op, i, inp))
                    # continue
                else:
                    previous_op = op_name_to_op[previous_op]

                if previous_op.name in constants:
                    """ Then this is a constant and we should use it to "bootstrap
                    
                    For example, we have  operation Unsqueeze whose inputs are "data" and "axes"
                    The "axes" is coming from a constant. Then we should assign the shape, type, and data of constant.
                    """
                    # print("Found constant: ", previous_op.name)
                    constant = constants[previous_op.name]
                    if constant.attribute and len(list(constant.attribute)) > 1 :
                        raise ValueError("Constant has more than one output. This is not supported.")
                    if not constant.attribute or len(list(constant.attribute)) == 0:
                        raise ValueError("Constant has no output. This is not supported.")
                    attribute = constant.attribute[0]
                    if not onnx_op.bootstrap_map:
                        onnx_op.bootstrap_map = {inp: attribute}
                    else:
                        onnx_op.bootstrap_map[inp] = attribute

                else:
                    source_index = previous_op.output.index(inp)
                    link_list.append(OnnxLink(
                        source=previous_op.name,
                        source_int=source_index,
                        var_name=inp,
                        sink_int=i,
                        sink=onnx_op.name))

        # For each output, if write_cache, then add write_to_file operation and add link from output
        for i, out in enumerate(onnx_op.output):
            if out in write_cache:
                write_cache.remove(out)
                write_to_file = create_write_to_file(out)
                if len(onnx_op.name.split("/")) > 1:
                    # Get the path
                    composite_path = "/".join(onnx_op.name.split("/")[:-1])
                    write_to_file.composite_path = composite_path
                    write_to_file.name = onnx_op.composite_path + "/" + write_to_file.name
                onnx_list.append(write_to_file)
                op_name_to_op[write_to_file.name] = onnx_op
                # Now add Link
                link = OnnxLink(
                    source=onnx_op.name,
                    source_int=i,
                    var_name=out,
                    sink_int=3,
                    sink=write_to_file.name
                )
                link_list.append(link)


        if include_tensor:
            relevant_tensors = dict()
            if onnx_op.output:
                relevant_tensors = {out: tensor_map[out] for out in onnx_op.output if out in tensor_map}

            if onnx_op.input:
                relevant_tensors.update({inp: tensor_map[inp] for inp in onnx_op.input if inp in tensor_map})

            onnx_op.tensor_map = relevant_tensors

        if onnx_op.opType == "Constant":
            # Then this will be a bootstrapped value for each place it is linked to.
            constants[onnx_op.name] = onnx_op
            continue
        onnx_list.append(onnx_op)


    # Now we need to determine which outputs are final outputs
    used_outs = {link.var_name for link in link_list}
    const_name = {const.output[0] for const in constants.values()}
    final_outputs = {out for out in output_to_op.keys() if
                     out not in used_outs and
                     out not in const_name and
                     len(out.split("/")) == 1}

    # Find the operation who produced those final outputs and add links.
    for i, out in enumerate(final_outputs):
        previous_op = output_to_op[out]
        if previous_op not in op_name_to_op:
            print(f"Could not find previous op: {previous_op}")
            raise ValueError("Could not find previous op: " + previous_op)
        else:
            previous_op = op_name_to_op[previous_op]

        source_index = previous_op.output.index(out)
        link_list.append(OnnxLink(
            source=previous_op.name,
            source_int=source_index,
            var_name=out,
            sink_int=i,
            sink=onnx_file))

    return OnnxGraph(
        name=onnx_file,
        inputs=unfilled_inputs,
        outputs=final_outputs,
        onnx_ops=onnx_list,
        onnx_links=link_list,
        parsed_onnx_file=parsed_onnx
    )


def onnx_folder_to_onnx_list(onnx_folder: str) -> List[OnnxGraph]:
    """Given folder of onnx files, convert to onnx list."""
    _onnx_list = []
    for file in os.listdir(onnx_folder):
        if file.endswith(".onnx"):
            onnx_graph = onnx_to_graph(os.path.join(f"{onnx_folder}/{file}"))
            if onnx_graph:
                _onnx_list.append(onnx_graph)

    return _onnx_list


def _calculate_composite_input_outputs(var_name: str, path1: str, path2: str, var_map: dict) -> None:
    """ Given two composite paths, calculate the inputs and outputs of the composite operations.

        This is done by looking at the links between the two composites and the operations in the two composites.

        Either the link is traveling downstream, upstream, or across stream.

        var_map should map inputs and outputs on each composite.

        :arg var_name: The variable name
        :arg path1: The first composite path
        :arg path2: The second composite path, not equal to path1
        :arg var_map: The variable map

        :returns: None
    """

    if not path1:
        # Going downstream from path1 to path2
        path2_split = path2.split("/")
        for i in range(1, len(path2_split)):
            join_back = "/".join(path2_split[:i+1])
            if "input" not in var_map[join_back]:
                var_map[join_back]["input"] = []

            if var_name not in var_map[join_back]["input"]:
                var_map[join_back]["input"].append(var_name)
        return

    if not path2:
        # Then we are going upstream from path1 to path2.
        path1_split = path1.split("/")
        for i in range(len(path1_split)):
            join_back = "/".join(path1_split[:i+1])
            if "output" not in var_map[join_back]:
                var_map[join_back]["output"] = []

            if var_name not in var_map[join_back]["output"]:
                var_map[join_back]["output"].append(var_name)
        return

    if path1.startswith(path2):
        """ e.g., /x/y/z and /x/y
            
            Then it is entirely upstream, because path1 is deeper than path2.
        """
        # Get the list that is just in path1 minus path2
        path1_split = path1.split("/")
        path2_split = path2.split("/")
        partial_split = path1_split[len(path2_split):]

        # Upstream means we are added outputs to each of the partial splits parts
        for i in range(len(partial_split)):
            join_back = path2 + "/" + "/".join(partial_split[:i + 1])
            if "output" not in var_map[join_back]:
                var_map[join_back]["output"] = []

            if var_name not in var_map[join_back]["output"]:
                var_map[join_back]["output"].append(var_name)
        return

    if path2.startswith(path1):
        # The other way around.
        path1_split = path1.split("/")
        path2_split = path2.split("/")
        partial_split = path2_split[len(path1_split):]

        # Downstream means we are adding inputs to each of the partial splits parts
        for i in range(len(partial_split)):
            join_back = path1 + "/" + "/".join(partial_split[:i + 1])

            if "input" not in var_map[join_back]:
                var_map[join_back]["input"] = []

            if var_name not in var_map[join_back]["input"]:
                var_map[join_back]["input"].append(var_name)
        return

    # If we're here, then there's a cross stream link.
    # For each composite on path in path1, needs output
    # For each composite on path in path2, needs input
    path1_split = path1.split("/")
    path2_split = path2.split("/")

    # Get the first n parts that are shared
    shared_parts = []
    for i in range(min(len(path1_split), len(path2_split))):
        if path1_split[:i+1] == path2_split[:i+1]:
            shared_parts.append("/".join(path1_split[:i+1]))
        else:
            break

    # For each shared path, we'll do nothing, for each unique on composite path 1, we need output and for each unique on composite path 2, we need input.

    for i in range(len(path1_split)):
        join_back = "/".join(path1_split[:i+1])
        if join_back in shared_parts:
            continue
        if "output" not in var_map[join_back]:
            var_map[join_back]["output"] = []
        if var_name not in var_map[join_back]["output"]:
            var_map[join_back]["output"].append(var_name)

    for i in range(len(path2_split)):
        join_back = "/".join(path2_split[:i+1])
        if join_back in shared_parts:
            continue
        if "input" not in var_map[join_back]:
            var_map[join_back]["input"] = []
        if var_name not in var_map[join_back]["input"]:
            var_map[join_back]["input"].append(var_name)

def _get_graphbook_type_from_str(type_str: str) -> graphbook.DataType:
    if "(int" in type_str or "(uint" in type_str:
        return graphbook.DataType.INTEGER
    elif "(float" in type_str or "(double" in type_str or "(bfloat" in type_str:
        return graphbook.DataType.DECIMAL
    elif "(string" in type_str:
        return graphbook.DataType.TEXT
    elif "(bool" in type_str:
        return graphbook.DataType.BOOLEAN
    else:
        return graphbook.DataType.NULL


def onnx_op_to_graphbook(onnx_op: OnnxOperation) -> graphbook.Operation:
    """ converts onnx operation to graphbook operation"""

    graphbook_inputs = []
    if onnx_op.input:
        for i, inp in enumerate(onnx_op.input):
            graphbook_var = graphbook.Variable(name=inp)
            if not onnx_op.op_type_meta_data:
                # Then it's our own read or write file
                if onnx_op.opType == "read_from_file":
                    if i == 0:
                        graphbook_var.primitive_name = "file_name"
                    elif i == 1:
                        graphbook_var.primitive_name = "dir_name"
                    elif i == 2:
                        graphbook_var.primitive_name = "extraction_schema"

            else:
                input_meta = list(onnx_op.op_type_meta_data["inputs"])
                if input_meta and i >= len(input_meta):
                    var_meta = dict(input_meta[0])
                    if "list" in var_meta:
                        # Then it's a list, for example for concat.
                        graphbook_var.primitive_name = f"list_item_{i}"
                else:
                    var_meta = dict(input_meta[i])
                    if 'name' in var_meta:
                        graphbook_var.primitive_name = var_meta['name']
            graphbook_inputs.append(graphbook_var)


    attribute_names = []
    if onnx_op.attribute:
        attribute_names = [attribute.name for attribute in onnx_op.attribute]

    if onnx_op.op_type_meta_data and "attribute" in onnx_op.op_type_meta_data:
        for i, attribute in onnx_op.op_type_meta_data["attribute"]:
            graphbook_var = graphbook.Variable(name=attribute["name"], primitive_name="attribute_" + attribute["name"])
            """ 
            Then we need to specify that it's "filled" in this operation
            This is because attributes in onnx act a bit like a conditional sometimes. 
            For example, for Constant operation, there is an attribute for each value type and shape it can take. 
            We add each attribute as an input and specify whether it is filled on this operation.
            Then later we can map based on the unique qualities of the operation and how it maps to graphbook.
            """
            graphbook_var.onnx_attribute = attribute["name"] in attribute_names

            if "type" in attribute:
                # This is a tensor
                graphbook_var.type = _get_graphbook_type_from_str(str(attribute['type']))
            elif attribute.i:
                graphbook_var.type = graphbook.DataType.INTEGER
            elif attribute.ints:
                graphbook_var.type = graphbook.DataType.INTEGER
                graphbook_var.shape = [len(attribute.ints)]

            graphbook_inputs.append(graphbook_var)

    graphbook_outputs = []
    if onnx_op.output:
        for i, out in enumerate(onnx_op.output):
            graphbook_var = graphbook.Variable(name=out)
            if not onnx_op.op_type_meta_data:
                # Then it's our own read or write file
                if onnx_op.opType == "write_to_file":
                    if i == 0:
                        graphbook_var.primitive_name = "file_name"
                    elif i == 1:
                        graphbook_var.primitive_name = "dir_name"
                    elif i == 2:
                        graphbook_var.primitive_name = "overwrite"
                    elif i == 3:
                        graphbook_var.primitive_name = "data"

            else:
                var_meta = onnx_op.op_type_meta_data["outputs"][i]
                if 'name' in var_meta:
                    graphbook_var.primitive_name = var_meta['name']

            graphbook_outputs.append(graphbook_var)

    # TODO: Add mapping here from onnx optype to graphbook schema type.
    return graphbook.Operation(
        name=onnx_op.name,
        primitive_name=onnx_op.opType,
        # For now, we won't say it's a primitive operation since it's not mapped yet to a real primitive.
        # type=graphbook.OperationType.PRIMITIVE_OPERATION,
        type=graphbook.OperationType.COMPOSITE_OPERATION, # This is temporary until they are properly mapped.
        inputs=graphbook_inputs,
        outputs=graphbook_outputs
    )


def onnx_graph_to_graphbook(onnx_graph: OnnxGraph) -> graphbook.Operation:
    """Converts onnx graph to Graphbook graph"""

    # First, get all the unique composite operations and assign the ops to the right operations.
    composite_names = {''}
    composite_map = defaultdict(list)
    name_to_op = {op.name: op for op in onnx_graph.onnx_ops}

    for op in onnx_graph.onnx_ops:
        if op.composite_path:
            split_path = op.composite_path.split("/")
            for i, part in enumerate(split_path):
                join_back = "/".join(split_path[:i+1])
                composite_names.add(join_back)

            composite_map[op.composite_path].append(op)
        else:
            composite_map[onnx_graph.name].append(op)

    composite_link_map = defaultdict(list)
    composite_var_map = defaultdict(dict)

    primitive_to_final_output_links = set()

    for link in onnx_graph.onnx_links:
        if link.sink == onnx_graph.name:
            # Then this is a final output
            composite_link_map[onnx_graph.name].append(link)
            _calculate_composite_input_outputs(
                var_name=link.var_name,
                path1=name_to_op[link.source].composite_path,
                path2="",
                var_map=composite_var_map)

        elif link.source in name_to_op and link.var_name in onnx_graph.outputs:
            # Then this is a final output
            primitive_to_final_output_links.add(link)

        elif link.sink in name_to_op:
            if name_to_op[link.sink].composite_path:
                composite_link_map[name_to_op[link.sink].composite_path].append(link)
                if link.source == onnx_graph.name:
                    _calculate_composite_input_outputs(
                        var_name=link.var_name,
                        path1="",
                        path2=name_to_op[link.sink].composite_path,
                        var_map=composite_var_map)
                elif not name_to_op[link.source].composite_path \
                        or name_to_op[link.sink].composite_path != name_to_op[link.source].composite_path:
                    # The link is traversing graph levels, so it should be an input to each composite along the path
                    _calculate_composite_input_outputs(
                        var_name=link.var_name,
                        path1=name_to_op[link.source].composite_path,
                        path2=name_to_op[link.sink].composite_path,
                        var_map=composite_var_map
                    )
            elif name_to_op[link.source].composite_path:
                # Then we are going from composite to non-composite
                composite_link_map[onnx_graph.name].append(link)
                _calculate_composite_input_outputs(
                    var_name=link.var_name,
                    path1=name_to_op[link.source].composite_path,
                    path2=name_to_op[link.sink].composite_path,
                    var_map=composite_var_map
                )
            else:
                composite_link_map[onnx_graph.name].append(link)
        else:
            # Then something fishy is happening. Why is this link not there?
            raise ValueError(f"Link sink {link.sink} not recognized")

    primitive_map = {}
    graphbook_composite_map = {}

    for name in composite_names:
        if not name:
            if onnx_graph.name in composite_map:
                # Then we've already been here.
                continue
            if name not in composite_map:
                composite_map[onnx_graph.name] = []
            else:
                composite_map[onnx_graph.name] = composite_map[name]

            name = onnx_graph.name
            inputs = list(onnx_graph.inputs)
            outputs = list(onnx_graph.outputs)
        else:
            if "input" not in composite_var_map[name]:
                composite_var_map[name]["input"] = []
            inputs = composite_var_map[name]["input"]

            if "output" not in composite_var_map[name]:
                composite_var_map[name]["output"] = []
            outputs = composite_var_map[name]["output"]

        this_primitive = {}
        if name in composite_map:
            this_primitive = {
                onnx_op.name: onnx_op_to_graphbook(onnx_op)
                for onnx_op in composite_map[name]
            }
            primitive_map.update(this_primitive)

        graphbook_composite_map[name] = graphbook.Operation(
            name=name,
            primitive_name=name,
            type=graphbook.OperationType.COMPOSITE_OPERATION,
            operations=list(this_primitive.values()),
            inputs=[graphbook.Variable(name=inp) for inp in inputs],
            outputs=[graphbook.Variable(name=out) for out in outputs],
            links=[]
        )

    # Links between composites only.
    for name in composite_names:
        if name == onnx_graph.name:
            continue
        split_name = name.split("/")
        if len(split_name) <= 1:
            continue

        # Then it is a sub-operation of some composite
        parent_name = "/".join(split_name[:-1])
        if len(parent_name) == 0:
            parent_name = onnx_graph.name

        sub_op = graphbook_composite_map[name]
        parent_composite = graphbook_composite_map[parent_name]
        parent_composite.operations.append(sub_op)

        for inp in sub_op.inputs:
            for comp_inp in parent_composite.inputs:
                if inp.name == comp_inp.name:
                    # add link
                    parent_composite.links.append(graphbook.Link(
                        source=graphbook.LinkEndpoint(operation="this", data=inp.name),
                        sink=graphbook.LinkEndpoint(operation=name, data=inp.name),
                    ))
        for out in sub_op.outputs:
            for comp_out in parent_composite.outputs:
                if out.name == comp_out.name:
                    parent_composite.links.append(graphbook.Link(
                        source=graphbook.LinkEndpoint(operation=name, data=out.name),
                        sink=graphbook.LinkEndpoint(operation="this", data=out.name),
                    ))

    # This is where links are connected between composites to primitives.
    for composite_name, link_list in composite_link_map.items():

        composite = graphbook_composite_map[composite_name]

        # For each link that ends in this composite graph, create a path of links from the source to here.
        for link in link_list:
            if link.source == onnx_graph.name:
                continue

            # Get the source and sink operations
            primitive_source = primitive_map[link.source]

            if primitive_source in composite.operations:
                # Then it's simply adding a link in this graph.
                composite.links.append(graphbook.Link(
                    source=graphbook.LinkEndpoint(operation=link.source, data=link.var_name),
                    sink=graphbook.LinkEndpoint(operation=link.sink, data=link.var_name),
                    var_name=link.var_name
                ))
            elif composite_name == onnx_graph.name:

                sink_name = "/".join(primitive_source.name.split("/")[:-1])
                next_composite = graphbook_composite_map[sink_name]
                #
                if link.var_name not in [out.name for out in next_composite.outputs]:
                    raise ValueError("Expected link var name to be in composite outputs")
                next_composite.links.append(graphbook.Link(
                    source=graphbook.LinkEndpoint(operation=primitive_source.name, data=link.var_name),
                    sink=graphbook.LinkEndpoint(operation="this", data=link.var_name),
                ))
            elif not primitive_source.name.startswith(composite_name):
                # If the primitive source is not from within this graph, it must be coming from parent graph.
                if link.var_name not in [inp.name for inp in composite.inputs]:
                    raise ValueError("Expected link var name to be in composite inputs")

                composite.links.append(graphbook.Link(
                    source=graphbook.LinkEndpoint(operation="this", data=link.var_name),
                    sink=graphbook.LinkEndpoint(operation=link.sink, data=link.var_name),
                ))

            # If it comes from within this graph, then we need to find the composite that produces it.
            else:
                # Get the next item in the path of primitive_source.name after composite_name
                split_name = primitive_source.name.split("/")
                split_composite_name = composite_name.split("/")
                if len(split_name) <= len(split_composite_name):
                    raise ValueError("Primitive source name is not longer than composite name")

                next_composite_name = "/".join(split_name[:len(split_composite_name)+1])
                next_composite = graphbook_composite_map[next_composite_name]

                if link.var_name not in [out.name for out in next_composite.outputs]:
                    raise ValueError("Expected link var name to be in composite outputs")
                composite.links.append(graphbook.Link(
                    source=graphbook.LinkEndpoint(operation=next_composite_name, data=link.var_name),
                    sink=graphbook.LinkEndpoint(operation=link.sink, data=link.var_name),
                ))

    return graphbook_composite_map[onnx_graph.name]



if __name__ == "__main__":

    onnx_list = onnx_folder_to_onnx_list("flan-t5-small-onnx")
    print('Generated onnx graphs, now converting to Graphbook')
    for graph in onnx_list:
        graphbook_root = onnx_graph_to_graphbook(graph)
        print("Generated: " + graphbook_root.name)