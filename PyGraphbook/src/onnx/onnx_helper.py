import json
from json import JSONEncoder
import numpy as np
import logging
import os

from contextlib import suppress
from typing import List, Dict, Optional

import onnx
from google.protobuf.json_format import MessageToJson
from netron.onnx import ModelFactory
# pylint: disable=protected-access
from netron.onnx import _Model as NetronModel
from onnx import numpy_helper

# from onnx.onnx_pb import TensorProto
from pydantic import BaseModel, Field, AliasChoices


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


# def _read_individual_weight(tensor: TensorProto, base_dir: str, out_dir: str):
#     pass

from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, TensorProto

class ExternalDataInfo:
    def __init__(self, tensor: TensorProto) -> None:
        self.location = ""
        self.offset = None
        self.length = None
        self.checksum = None
        self.basepath = ""

        for entry in tensor.external_data:
            setattr(self, entry.key, entry.value)

        if self.offset:
            self.offset = int(self.offset)

        if self.length:
            self.length = int(self.length)

def convert_onnx_to_dict(onnx_file_name: str, load_data: bool = False) -> ParsedOnnxFile:
    """ Convert onnx file to dict."""

    onnx_model = onnx.load_model(onnx_file_name, load_external_data=False)
    # onnx.checker.check_model(onnx_file_name)

    # folder_name = "/".join(onnx_file_name.split("/")[:-1])
    # file_name = onnx_file_name.split("/")[-1]
    needs_load = False

    tensor_map = {}
    # for file in os.listdir(os.path.dirname(onnx_file_name + "/"))
    for tensor in onnx_model.graph.initializer:
        tensor_map[tensor.name] = []
        if load_data:
            # tensor_map[tensor.name] = (
            logging.debug(f"Reading tensor: {tensor.name} from file {onnx_file_name}")

            try:
                read_onnx_data(onnx_file_name, tensor, do_write=True)
            except Exception as e:
                if needs_load:
                    raise e
                needs_load = True
                onnx_model = onnx.load_model(onnx_file_name, load_external_data=True)
                break

    if needs_load:
        for tensor in onnx_model.graph.initializer:
            tensor_map[tensor.name] = numpy_helper.to_array(tensor)

    # exit(1)
    logging.debug('finished reading netron model')

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
    dims: List[str] | List[int] = Field(None, alias="dims")
    dataType: int = Field(None, alias="dataType")
    data_type: str = Field(None, alias="data_type")
    value: int | List[int] = Field(None, alias="value")


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
        'input': [dir_name, split_name[-1], "{}"],
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
        'input': [dir_name, split_name[-1], "true", file_name],
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
    if not isinstance(onnx_tensor.dataType, int) \
            or not isinstance(type_list, list) \
            or onnx_tensor.dataType - 1 >= len(type_list):
        raise ValueError("Invalid data type for onnx tensor: " + str(onnx_tensor))

    return type_list[onnx_tensor.dataType - 1]


def onnx_to_graph(onnx_file: str, load_data: bool = False) -> Optional[OnnxGraph]:
    """ Convert given onnx file to list of OnnxOperation objects. and tensormap."""

    logging.info(f"Converting {onnx_file} to NetronModel")
    try:
        parsed_onnx = convert_onnx_to_dict(onnx_file, load_data=load_data)
    except Exception as e:
        print(f"Could not convert {onnx_file} to NetronModel. {e}")
        return None

    logging.info(f"Converting {onnx_file} to OnnxGraph")

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
    op_name_to_op = {}
    for op in onnx_json["graph"]["node"]:
        onnx_op = OnnxOperation(**op)
        # onnx_op.name = onnx_op.name
        onnx_op.op_type_meta_data = dict(parsed_onnx.netron_model.metadata.metadata[onnx_op.opType])
        if len(onnx_op.name.split("/")) > 1:
            onnx_op.composite_path = "/".join(onnx_op.name.split("/")[:-1])

        if onnx_op.attribute:
            for attribute in onnx_op.attribute:
                if attribute.t and attribute.t.dataType:
                    data_type_str = _fetch_tensor_type_for_onnx_tensor(
                        parsed_onnx.netron_model,
                        onnx_op.opType,
                        attribute.t)
                    attribute.t.data_type = data_type_str
                elif attribute.i:
                    attribute.t = OnnxTensor(
                        dims=[],
                        data_type="tensor(int32)",
                        value=int(attribute.i)
                    )
                elif attribute.ints:
                    attribute.t = OnnxTensor(
                        dims=[len(attribute.ints)],
                        data_type="tensor(int32)",
                        value=[int(i) for i in list(attribute.ints)]
                    )

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
                        source_int=len(unfilled_inputs) - 1,
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
                    if constant.attribute and len(list(constant.attribute)) > 1:
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


def onnx_folder_to_onnx_list(onnx_folder: str, load_data: bool=False) -> List[OnnxGraph]:
    """Given folder of onnx files, convert to onnx list."""
    _onnx_list = []
    for file in os.listdir(onnx_folder):
        if file.endswith(".onnx"):
            onnx_graph = onnx_to_graph(os.path.join(f"{onnx_folder}/{file}"), load_data=load_data)
            if onnx_graph:
                _onnx_list.append(onnx_graph)

    return _onnx_list


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def numpy_to_json(numpy_data: np.ndarray):
    return json.dumps(numpy_data, cls=NumpyArrayEncoder)  # use dump() to write array into file


def read_onnx_data(onnx_file_name: str, tensor: TensorProto, do_write: bool=True):
    """ Read onnx data from external data file."""

    info = ExternalDataInfo(tensor)
    file_name = info.location.lstrip("/.")
    base_dir = os.path.dirname(onnx_file_name)
    external_data_file_path = os.path.join(base_dir, file_name)
    with open(external_data_file_path, "rb") as data_file:
        if info.offset:
            data_file.seek(info.offset)

        if info.length:
            tensor.raw_data = data_file.read(info.length)
        else:
            tensor.raw_data = data_file.read()

    tensor_numpy = numpy_helper.to_array(tensor, base_dir=base_dir)
    if do_write:
        as_json = numpy_to_json(tensor_numpy)
        weight_dir = f"{onnx_file_name}_weights"
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        with open(f"{weight_dir}/{tensor.name}.json", 'w') as f:
            f.write(as_json)

    return tensor_numpy
