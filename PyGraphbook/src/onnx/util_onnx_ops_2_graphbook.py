from typing import List
import os
import json
import logging
import copy
from src import graph as graphbook
from src.onnx.onnx_helper import OnnxOperation


def _produce_mapping() -> dict:
    """ Produces the mapping from onnx op to graphbook. """

    with open(os.path.join(os.path.dirname(__file__), "op_mappings.json")) as f:
        op_gb_mapping = json.load(f)

    onnx_to_mapping = {}
    for item in op_gb_mapping["op_mappings"]:
        op_map = item["op_name_map"]
        onnx_to_mapping[op_map["onnx_name"]] = item

    return onnx_to_mapping


def _fetch_schema_lib() -> dict:
    """ Fetches the schema library. """
    folder = os.path.join(os.path.dirname(__file__), "../../../compute_operations/")
    gb_op_dict = {}
    for sub_folder in os.listdir(folder):
        for file in os.listdir(os.path.join(folder, sub_folder)):
            if file.endswith(".json"):
                with open(os.path.join(folder, sub_folder, file), "r") as f:
                    graph_json = json.load(f)
                    graph_obj = graphbook.Operation.model_validate(graph_json)
                    gb_op_dict[graph_obj.name] = graph_obj

    return gb_op_dict


# Read mapping
OP_GB_MAPPING = _produce_mapping()
GB_OPS = _fetch_schema_lib()

MULTI_DIM_BROADCASTING_ONNX_OPS = [
    "Add",
    "And",
    "Div",
    "Equal",
    "Greater",
    "Less",
    "Max",
    "Mean",
    "Min",
    "Mul",
    "Or",
    "Pow",
    "Sub",
    "Sum",
    "Where",
    "Xor"
]

def _add_multi_dim_broadcasting(gb_op: graphbook.Operation):
    """
    Add multi dimensionional broadcasting rules to second input on graphbook operation.

    In ONNX, a set of tensors are multidirectional broadcastable to the same shape if one of the following is true:

    The tensors all have exactly the same shape.
    The tensors all have the same number of dimensions and the length of each dimensions is either a common length or 1.
    The tensors that have too few dimensions can have their shapes prepended with a dimension of length 1 to satisfy property 2.
    For example, the following tensor shapes are supported by multidirectional broadcasting:

    shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar ==> shape(result) = (2, 3, 4, 5)
    shape(A) = (2, 3, 4, 5), shape(B) = (5,), ==> shape(result) = (2, 3, 4, 5)
    shape(A) = (4, 5), shape(B) = (2, 3, 4, 5), ==> shape(result) = (2, 3, 4, 5)
    shape(A) = (1, 4, 5), shape(B) = (2, 3, 1, 1), ==> shape(result) = (2, 3, 4, 5)
    shape(A) = (3, 4, 5), shape(B) = (2, 1, 1, 1), ==> shape(result) = (2, 3, 4, 5)

    """
    if gb_op.primitive_name not in MULTI_DIM_BROADCASTING_ONNX_OPS:
        return

    get_shape = _copy_primitive("get_shape")
    get_size = _copy_primitive("get_dimension_size")



def _copy_primitive(primitive_name: str):
    gb_copy = copy.copy(GB_OPS[primitive_name])
    gb_copy.inputs = [copy.copy(inp) for inp in gb_copy.inputs]
    gb_copy.outputs = [copy.copy(out) for out in gb_copy.outputs]
    gb_copy.assertions = []

    return gb_copy


class Base:
    # simply allow additional args in base class
    def convert(
            self,
            onnx_op: OnnxOperation,
            graphbook_inputs: List[graphbook.Variable],
            graphbook_outputs: List[graphbook.Variable]) -> graphbook.Operation:
        pass


def _convert_from_primitive(
        graphbook_inputs: List[graphbook.Variable],
        graphbook_outputs: List[graphbook.Variable],
        gb_op: graphbook.Operation,
        gb_map: dict) -> graphbook.Operation:
    """ Convert ONNX primitive operation to Graphbook operation. """

    input_value_dict = {val["onnx_name"]: val["gb_name"] for val in gb_map["inputs"]}
    output_value_dict = {val["onnx_name"]: val["gb_name"] for val in gb_map["outputs"]}
    default_dict = {val["gb_name"]: val["default"] for val in gb_map["inputs"] if "default" in val}

    if gb_op.type != graphbook.OperationType.PRIMITIVE_OPERATION and gb_op.name not in ["Multiply by Negative One"]:
        gb_input_value_dict = {val.name: val for val in gb_op.inputs}
        gb_output_value_dict = {val.name: val for val in gb_op.outputs}
    else:
        gb_input_value_dict = {val.primitive_name: val for val in gb_op.inputs}
        gb_output_value_dict = {val.primitive_name: val for val in gb_op.outputs}

    for onnx_input in graphbook_inputs:
        if onnx_input.primitive_name not in input_value_dict:
            continue

        gb_input_name = input_value_dict[onnx_input.primitive_name]
        if gb_input_name not in gb_input_value_dict:
            logging.info(f"Input {gb_input_name} not found in {gb_op.name}")
            raise ValueError(f"Input {gb_input_name} not found in {gb_op.name}")

        gb_var = gb_input_value_dict[gb_input_name]

        if onnx_input.data:
            gb_var.data = onnx_input.data
            gb_var.shape = onnx_input.shape
            gb_var.type = onnx_input.type
        elif gb_input_name in default_dict:
            gb_var.data = default_dict[gb_input_name]
            gb_var.shape = []
            if type(gb_var.data) == str:
                gb_var.type = graphbook.DataType.TEXT
            elif type(gb_var.data) == int:
                gb_var.type = graphbook.DataType.INTEGER
            elif type(gb_var.data) == bool:
                gb_var.type = graphbook.DataType.BOOLEAN
            elif type(gb_var.data) == float:
                gb_var.type = graphbook.DataType.DECIMAL
            else:
                gb_var.type = graphbook.DataType.TEXT

        gb_var.name = onnx_input.name

    for onnx_output in graphbook_outputs:
        if onnx_output.primitive_name not in output_value_dict:
            continue

        gb_output_name = output_value_dict[onnx_output.primitive_name]
        if gb_output_name not in gb_output_value_dict:
            raise ValueError(f"Output {gb_output_name} not found in {gb_op.name}")

        gb_var = gb_output_value_dict[gb_output_name]
        gb_var.name = onnx_output.name

    return gb_op


class Cast(Base):
    def convert(
            self,
            onnx_op: OnnxOperation,
            graphbook_inputs: List[graphbook.Variable],
            graphbook_outputs: List[graphbook.Variable]) -> graphbook.Operation:
        """ Convert ONNX cast operation to Graphbook operation. """

        data_type_as_int = graphbook_inputs[-1].data
        if data_type_as_int in [1, 10, 11]:
            primitive_name = "parse_decimal"
            # Float
        elif data_type_as_int == 8:
            primitive_name = "parse_text"
            # Float
        elif data_type_as_int == 9:
            # Text
            primitive_name = "parse_boolean"
        else:
            primitive_name = "parse_integer"

        gb_op = _copy_primitive(primitive_name)

        gb_map = OP_GB_MAPPING[onnx_op.opType]
        gb_op = _convert_from_primitive(graphbook_inputs, graphbook_outputs, gb_op, gb_map)
        gb_op.name = onnx_op.name
        return gb_op


class Shape(Base):

    def convert(
            self,
            onnx_op: OnnxOperation,
            graphbook_inputs: List[graphbook.Variable],
            graphbook_outputs: List[graphbook.Variable]) -> graphbook.Operation:
        """ Convert ONNX shape operation to Graphbook operation. """

        shape_op = _copy_primitive("get_shape")
        shape_op_map = OP_GB_MAPPING[onnx_op.opType]
        shape_gb_op = _convert_from_primitive(graphbook_inputs, graphbook_outputs, shape_op, shape_op_map)

        # Determine if we need to slice
        if len(graphbook_inputs) != 3:
            shape_gb_op.name = onnx_op.name
            return shape_gb_op

        end = graphbook_inputs[1].data
        start = graphbook_inputs[2].data

        if start is None and end is None:
            shape_gb_op.name = onnx_op.name
            return shape_gb_op

        # Otherwise, we're slicing a 1D array
        slice_op = _copy_primitive("slice")

        slice_op.name = onnx_op.name + "_wslice"

        slice_op.inputs[1].data = 0
        slice_op.inputs[1].type = graphbook.DataType.INTEGER
        slice_op.inputs[1].shape = []

        slice_op.inputs[2].type = graphbook.DataType.INTEGER
        slice_op.inputs[2].shape = []
        slice_op.inputs[3].type = graphbook.DataType.INTEGER
        slice_op.inputs[3].shape = []

        if start is not None:
            slice_op.inputs[2].data = start
        else:
            slice_op.inputs[2].data = 0

        if end is not None:
            if end > 0:
                slice_op.inputs[3].data = end
            else:
                # TODO, add a get_dimension_size and subtract 1.
                pass
        else:
            # TODO, add a get_dimension_size and subtract 1.
            # slice_op.inputs[3].data =
            pass

        slice_op.outputs[0].name = graphbook_outputs[0].name

        link = graphbook.Link(
            source=graphbook.LinkEndpoint(operation=shape_gb_op.name, data=shape_gb_op.outputs[0].name),
            sink=graphbook.LinkEndpoint(operation=slice_op.name, data=slice_op.inputs[0].name)
        )

        # Create composite with shape and slice and link
        return graphbook.Operation(
            name=onnx_op.name,
            primitive_name=str(onnx_op.opType),
            assertions=[],
            type=graphbook.OperationType.COMPOSITE_OPERATION,
            inputs=copy.copy(shape_gb_op.inputs),
            outputs=copy.copy(slice_op.outputs),
            operations=[shape_gb_op, slice_op],
            links=[
                link,
                graphbook.Link(
                   source=graphbook.LinkEndpoint(operation="this", data=shape_gb_op.inputs[0].name),
                   sink=graphbook.LinkEndpoint(operation=shape_gb_op.name, data=shape_gb_op.inputs[0].name)
                ),
                graphbook.Link(
                    source=graphbook.LinkEndpoint(operation=slice_op.name, data=slice_op.outputs[0].name),
                    sink=graphbook.LinkEndpoint(operation="this", data=slice_op.outputs[0].name)
                )
            ]
        )


class Unsqueeze(Base):
    def convert(
            self,
            onnx_op: OnnxOperation,
            graphbook_inputs: List[graphbook.Variable],
            graphbook_outputs: List[graphbook.Variable]) -> graphbook.Operation:
        """ Convert ONNX unsqueeze operation to Graphbook operation. """

        gb_op = _copy_primitive("expand_one_dimension")
        gb_op.inputs[0].name = graphbook_inputs[0].name
        gb_op.inputs[1].name = graphbook_inputs[1].name
        if graphbook_inputs[1].data:
            try:
                gb_op.inputs[1].data = graphbook_inputs[1].data[0]
            except IndexError:
                raise ValueError(f"Invalid data for unsqueeze {graphbook_inputs[1].data}")

            gb_op.inputs[1].shape = []
            gb_op.inputs[1].type = graphbook.DataType.INTEGER

        gb_op.outputs[0].name = graphbook_outputs[0].name

        gb_op.name = onnx_op.name

        return gb_op


class Concat(Base):

    def create_op(self):
        """ Create a copy of the concat operation. """
        return _copy_primitive("concatenate")

    def convert(
            self,
            onnx_op: OnnxOperation,
            graphbook_inputs: List[graphbook.Variable],
            graphbook_outputs: List[graphbook.Variable]) -> graphbook.Operation:
        """ Convert ONNX concat operation to Graphbook operation. """

        if len(graphbook_inputs) == 3:
            # Then this is gonna be just regular primitive concat
            gb_op = self.create_op()
            gb_op.name = onnx_op.name
            gb_op.inputs[0].name = graphbook_inputs[0].name
            gb_op.inputs[1].name = graphbook_inputs[1].name
            gb_op.inputs[2].name = graphbook_inputs[2].name

            for i, inp in enumerate(graphbook_inputs):
                if inp.data:
                    gb_op.inputs[i].data = inp.data
                    gb_op.inputs[i].shape = inp.shape
                    gb_op.inputs[i].type = inp.type
                    break

            if gb_op.inputs[2].data is None:
                # then check for default
                gb_op.inputs[2].data = 1

            gb_op.outputs[0].name = graphbook_outputs[0].name
            return gb_op

        # Otherwise, we're concatenating multiple times

        ops = []
        links = []
        for i, inp in enumerate(graphbook_inputs):
            if i == 0:
                continue
            if i == len(graphbook_inputs) - 1:
                break

            gb_op = self.create_op()
            gb_op.name = onnx_op.name + f"_{i}"
            if len(ops) > 0:
                gb_op.inputs[0].name = ops[-1].outputs[0].name
            else:
                gb_op.inputs[0].name = graphbook_inputs[0].name

            gb_op.inputs[1].name = graphbook_inputs[i].name

            # Link for the axis
            links.append(
                # Link to the axis.
                graphbook.Link(
                    source=graphbook.LinkEndpoint(operation="this", data="axis"),
                    sink=graphbook.LinkEndpoint(operation=gb_op.name, data="dimension_index")
                )
            )
            # Link for the right concatenate
            links.append(
                graphbook.Link(
                    source=graphbook.LinkEndpoint(operation="this", data=gb_op.inputs[1].name),
                    sink=graphbook.LinkEndpoint(operation=gb_op.name, data=gb_op.inputs[1].name)
                )
            )
            if len(ops) > 0:
                links.append(graphbook.Link(
                    source=graphbook.LinkEndpoint(operation=ops[-1].name, data=ops[-1].outputs[0].name),
                    sink=graphbook.LinkEndpoint(operation=gb_op.name, data=gb_op.inputs[1].name)
                ))
            else:
                links.append(graphbook.Link(
                    source=graphbook.LinkEndpoint(operation="this", data=gb_op.inputs[0].name),
                    sink=graphbook.LinkEndpoint(operation=gb_op.name, data=gb_op.inputs[0].name)
                ))

            ops.append(gb_op)

        links.append(graphbook.Link(
            source=graphbook.LinkEndpoint(operation=ops[-1].name, data=ops[-1].outputs[0].name),
            sink=graphbook.LinkEndpoint(operation="this", data=graphbook_outputs[0].name)
        ))

        comp_inputs = [copy.copy(inp) for inp in graphbook_inputs]

        if comp_inputs[-1].data is None:
            # then check for default
            comp_inputs[-1].data = 1

        return graphbook.Operation(
            name=onnx_op.name,
            primitive_name=str(onnx_op.opType),
            assertions=[],
            type=graphbook.OperationType.COMPOSITE_OPERATION,
            inputs=comp_inputs,
            outputs=graphbook_outputs,
            operations=ops,
            links=links
        )


class Tile(Base):
    def convert(
            self,
            onnx_op: OnnxOperation,
            graphbook_inputs: List[graphbook.Variable],
            graphbook_outputs: List[graphbook.Variable]) -> graphbook.Operation:
        """ Convert ONNX tile operation to Graphbook operation. """

        broadcast_to_shape = _copy_primitive("broadcast_to_shape")
        get_shape = _copy_primitive("get_shape")
        _copy_primitive("multiply")
        pass


class Sqrt(Base):
    def convert(
            self,
            onnx_op: OnnxOperation,
            graphbook_inputs: List[graphbook.Variable],
            graphbook_outputs: List[graphbook.Variable]) -> graphbook.Operation:
        """ Convert ONNX sqrt operation to Graphbook operation. """

        get_shape = _copy_primitive("get_shape")
        broadcast = _copy_primitive("broadcast_to_shape")
        sqrt = _copy_primitive("element_wise_exponentiate")

        # Assign target
        broadcast.inputs[0].data = 0.5
        broadcast.inputs[0].shape = []
        broadcast.inputs[0].type = graphbook.DataType.DECIMAL

        links = [
            graphbook.Link(
                source=graphbook.LinkEndpoint(operation="this", data=graphbook_inputs[0].name),
                sink=graphbook.LinkEndpoint(operation=get_shape.name, data=get_shape.inputs[0].name)
            ),
            graphbook.Link(
                source=graphbook.LinkEndpoint(operation=get_shape.name, data=get_shape.outputs[0].name),
                sink=graphbook.LinkEndpoint(operation=broadcast.name, data=broadcast.inputs[1].name)
            ),
            graphbook.Link(
                source=graphbook.LinkEndpoint(operation=broadcast.name, data=broadcast.outputs[0].name),
                sink=graphbook.LinkEndpoint(operation=sqrt.name, data=sqrt.inputs[1].name)
            ),
            graphbook.Link(
                source=graphbook.LinkEndpoint(operation="this", data=graphbook_outputs[0].name),
                sink=graphbook.LinkEndpoint(operation=sqrt.name, data=sqrt.outputs[0].name)
            ),
            graphbook.Link(
                source=graphbook.LinkEndpoint(operation=sqrt.name, data=sqrt.outputs[0].name),
                sink=graphbook.LinkEndpoint(operation="this", data=graphbook_outputs[0].name)
            )
        ]

        return graphbook.Operation(
            name=onnx_op.name,
            primitive_name=str(onnx_op.opType),
            assertions=[],
            type=graphbook.OperationType.COMPOSITE_OPERATION,
            inputs=copy.copy(graphbook_inputs),
            outputs=copy.copy(graphbook_outputs),
            operations=[get_shape, broadcast, sqrt],
            links=links
        )


class Min(Base):

    def create_min_0th(self):
        """ Create a copy of the minimum operation. """
        minimum = _copy_primitive("minimum")
        minimum.inputs[1].data = 0
        minimum.inputs[1].shape = []
        minimum.inputs[1].type = graphbook.DataType.INTEGER
        minimum.inputs[2].data = False
        minimum.inputs[2].shape = []
        minimum.inputs[2].type = graphbook.DataType.BOOLEAN
        return minimum


    def create_expand_0th(self):
        """ Create a copy of the expand operation. """
        expand_op = _copy_primitive("expand_one_dimension")
        expand_op.inputs[1].data = 0
        expand_op.inputs[1].shape = []
        expand_op.inputs[1].type = graphbook.DataType.INTEGER
        return expand_op

    def create_concat_0th(self):
        """ Create a copy of the concat operation. """
        concat = _copy_primitive("concatenate")
        concat.inputs[1].data = 0
        concat.inputs[1].shape = []
        concat.inputs[1].type = graphbook.DataType.INTEGER
        return concat

    def convert(
            self,
            onnx_op: OnnxOperation,
            graphbook_inputs: List[graphbook.Variable],
            graphbook_outputs: List[graphbook.Variable]) -> graphbook.Operation:
        """ Convert ONNX minimum operation to Graphbook operation. """

        last_concat_output = None
        last_expand_output = None
        ops = []
        links = []
        for i, inp in enumerate(graphbook_inputs):

            expand = self.create_expand_0th()
            expand.name = onnx_op.name + f"_expand_{i}"

            links.append(
                graphbook.Link(
                    source=graphbook.LinkEndpoint(operation="this", data=inp.name),
                    sink=graphbook.LinkEndpoint(operation=expand.name, data=expand.inputs[0].name)
                )
            )

            ops.append(expand)

            if i == 0:
                # On first iteration, just use last expand.
                last_expand_output = expand
                continue

            concat = self.create_concat_0th()
            concat.name = onnx_op.name + f"_concat_{i}"
            ops.append(concat)

            # Always a link from this expand to this concat.
            links.append(
                graphbook.Link(
                    source=graphbook.LinkEndpoint(operation=expand.name, data=expand.outputs[0].name),
                    sink=graphbook.LinkEndpoint(operation=concat.name, data=concat.inputs[1].name)
                )
            )

            # Link from last concat and this expand to concat
            # If no last concat, then use last expand

            if last_concat_output is not None:
                links.append(
                    graphbook.Link(
                        source=graphbook.LinkEndpoint(operation=last_concat_output.name, data=last_concat_output.outputs[0].name),
                        sink=graphbook.LinkEndpoint(operation=concat.name, data=concat.inputs[0].name)
                    )
                )
            else:
                # Link comes from last expand
                links.append(
                    graphbook.Link(
                        source=graphbook.LinkEndpoint(operation=last_expand_output.name, data=last_expand_output.outputs[0].name),
                        sink=graphbook.LinkEndpoint(operation=concat.name, data=concat.inputs[0].name)
                    )
                )

            last_concat_output = concat

        # Create the minimum operation
        minimum = self.create_min_0th()
        minimum.name = onnx_op.name + "_min"
        ops.append(minimum)

        # Link from last concat to the minimum
        links.append(
            graphbook.Link(
                source=graphbook.LinkEndpoint(operation=last_concat_output.name, data=last_concat_output.outputs[0].name),
                sink=graphbook.LinkEndpoint(operation=minimum.name, data=minimum.inputs[0].name)
            )
        )

        # Link from minimum output to "this"
        links.append(
            graphbook.Link(
                source=graphbook.LinkEndpoint(operation=minimum.name, data=minimum.outputs[0].name),
                sink=graphbook.LinkEndpoint(operation="this", data=graphbook_outputs[0].name)
            )
        )

        return graphbook.Operation(
            name=onnx_op.name,
            primitive_name=str(onnx_op.opType),
            assertions=[],
            type=graphbook.OperationType.COMPOSITE_OPERATION,
            inputs=graphbook_inputs,
            outputs=graphbook_outputs,
            operations=ops,
            links=links
        )


class Max(Base):

    def create_max_0th(self):
        """ Create a copy of the maximum operation. """
        minimum = _copy_primitive("maximum")
        minimum.inputs[1].data = 0
        minimum.inputs[1].shape = []
        minimum.inputs[1].type = graphbook.DataType.INTEGER
        minimum.inputs[2].data = False
        minimum.inputs[2].shape = []
        minimum.inputs[2].type = graphbook.DataType.BOOLEAN
        return minimum


    def create_expand_0th(self):
        """ Create a copy of the expand operation. """
        expand_op = _copy_primitive("expand_one_dimension")
        expand_op.inputs[1].data = 0
        expand_op.inputs[1].shape = []
        expand_op.inputs[1].type = graphbook.DataType.INTEGER
        return expand_op

    def create_concat_0th(self):
        """ Create a copy of the concat operation. """
        concat = _copy_primitive("concatenate")
        concat.inputs[1].data = 0
        concat.inputs[1].shape = []
        concat.inputs[1].type = graphbook.DataType.INTEGER
        return concat

    def convert(
            self,
            onnx_op: OnnxOperation,
            graphbook_inputs: List[graphbook.Variable],
            graphbook_outputs: List[graphbook.Variable]) -> graphbook.Operation:
        """ Convert ONNX maximum operation to Graphbook operation. """

        last_concat_output = None
        last_expand_output = None
        ops = []
        links = []
        for i, inp in enumerate(graphbook_inputs):

            expand = self.create_expand_0th()
            expand.name = onnx_op.name + f"_expand_{i}"

            links.append(
                graphbook.Link(
                    source=graphbook.LinkEndpoint(operation="this", data=inp.name),
                    sink=graphbook.LinkEndpoint(operation=expand.name, data=expand.inputs[0].name)
                )
            )

            ops.append(expand)

            if i == 0:
                # On first iteration, just use last expand.
                last_expand_output = expand
                continue

            concat = self.create_concat_0th()
            concat.name = onnx_op.name + f"_concat_{i}"
            ops.append(concat)

            # Always a link from this expand to this concat.
            links.append(
                graphbook.Link(
                    source=graphbook.LinkEndpoint(operation=expand.name, data=expand.outputs[0].name),
                    sink=graphbook.LinkEndpoint(operation=concat.name, data=concat.inputs[1].name)
                )
            )

            # Link from last concat and this expand to concat
            # If no last concat, then use last expand

            if last_concat_output is not None:
                links.append(
                    graphbook.Link(
                        source=graphbook.LinkEndpoint(operation=last_concat_output.name, data=last_concat_output.outputs[0].name),
                        sink=graphbook.LinkEndpoint(operation=concat.name, data=concat.inputs[0].name)
                    )
                )
            else:
                # Link comes from last expand
                links.append(
                    graphbook.Link(
                        source=graphbook.LinkEndpoint(operation=last_expand_output.name, data=last_expand_output.outputs[0].name),
                        sink=graphbook.LinkEndpoint(operation=concat.name, data=concat.inputs[0].name)
                    )
                )

            last_concat_output = concat

        # Create the minimum operation
        minimum = self.create_max_0th()
        minimum.name = onnx_op.name + "_max"
        ops.append(minimum)

        # Link from last concat to the minimum
        links.append(
            graphbook.Link(
                source=graphbook.LinkEndpoint(operation=last_concat_output.name, data=last_concat_output.outputs[0].name),
                sink=graphbook.LinkEndpoint(operation=minimum.name, data=minimum.inputs[0].name)
            )
        )

        # Link from minimum output to "this"
        links.append(
            graphbook.Link(
                source=graphbook.LinkEndpoint(operation=minimum.name, data=minimum.outputs[0].name),
                sink=graphbook.LinkEndpoint(operation="this", data=graphbook_outputs[0].name)
            )
        )

        return graphbook.Operation(
            name=onnx_op.name,
            primitive_name=str(onnx_op.opType),
            assertions=[],
            type=graphbook.OperationType.COMPOSITE_OPERATION,
            inputs=graphbook_inputs,
            outputs=graphbook_outputs,
            operations=ops,
            links=links
        )


class Transpose(Base):
    def convert(
            self,
            onnx_op: OnnxOperation,
            graphbook_inputs: List[graphbook.Variable],
            graphbook_outputs: List[graphbook.Variable]) -> graphbook.Operation:
        """ Convert ONNX transpose operation to Graphbook operation. """

        transpose = _copy_primitive("swap_dimensions")
        transpose.name = onnx_op.name

        transpose.inputs[0].name = graphbook_inputs[0].name
        transpose.inputs[0].data = graphbook_inputs[0].data

        if graphbook_inputs[1].data is None:
            # Then there's no swap dimensions set, and the default is to reverse the dimensions
            graphbook_inputs[1].data = [1, 0]

        # Determine which dimensions to swap
        dimensions = graphbook_inputs[1].data
        """
            [0, 2, 3, 1] corresponds to swapping 1, 3
            [0, 2, 1, 3] corresponds to swapping 1, 2
            [0, 3, 1, 2] corresponds to swapping 3, 1
        """
        dims_to_swap = []
        for i, dim in enumerate(dimensions):
            if dim < 0:
                raise ValueError(f"Invalid dimension {dim} for transpose {onnx_op.name}")
            if dim == i:
                continue

            # Example: i = 1 and dim = 2
            # Then we want to know where dim = 1 is and that index is the other swap
            dims_to_swap = [i, dimensions.index(i)]
            break

        transpose.inputs[1].data = dims_to_swap[0]
        transpose.inputs[1].shape = []
        transpose.inputs[1].type = graphbook.DataType.INTEGER
        transpose.inputs[2].data = dims_to_swap[1]
        transpose.inputs[2].shape = []
        transpose.inputs[2].type = graphbook.DataType.INTEGER

        transpose.outputs[0].name = graphbook_outputs[0].name
        return transpose


class ReadFromFile(Base):
    def convert(
            self,
            onnx_op: OnnxOperation,
            graphbook_inputs: List[graphbook.Variable],
            graphbook_outputs: List[graphbook.Variable]) -> graphbook.Operation:
        """ Convert read from file operation to Graphbook operation. """

        op = _convert_from_primitive(
            graphbook_inputs,
            graphbook_outputs,
            _copy_primitive("read_from_file"),
            OP_GB_MAPPING[onnx_op.opType])

        op.name = onnx_op.name

        # Convert name over.
        op.outputs[0].name = graphbook_outputs[0].name
        return op


class WriteToFile(Base):
    def convert(
            self,
            onnx_op: OnnxOperation,
            graphbook_inputs: List[graphbook.Variable],
            graphbook_outputs: List[graphbook.Variable]) -> graphbook.Operation:
        """ Convert read from file operation to Graphbook operation. """

        op = _convert_from_primitive(
            graphbook_inputs,
            graphbook_outputs,
            _copy_primitive("write_to_file"),
            OP_GB_MAPPING[onnx_op.opType])

        op.name = onnx_op.name

        return op


class Softmax(Base):
    def convert(
            self,
            onnx_op: OnnxOperation,
            graphbook_inputs: List[graphbook.Variable],
            graphbook_outputs: List[graphbook.Variable]) -> graphbook.Operation:
        """ Convert softmax operation to Graphbook operation. """

        softmax = _copy_primitive("Softmax")
        softmax.inputs[1].primitive_name = softmax.inputs[1].name

        softmax_op = _convert_from_primitive(
            graphbook_inputs,
            graphbook_outputs,
            softmax,
            OP_GB_MAPPING[onnx_op.opType])

        softmax_op.name = onnx_op.name

        return softmax_op


OP_MAPPING = {
    "Cast": Cast(),
    "Shape": Shape(),
    "Unsqueeze": Unsqueeze(),
    "Softmax": Softmax(),
    "Concat": Concat(),
    "Sqrt": Sqrt(),
    # "Tile": Tile(),
    "Min": Min(),
    "Max": Max(),
    "Transpose": Transpose(),
    "read_from_file": ReadFromFile(),
    "write_to_file": WriteToFile()
}


def onnx_to_graphbook(
        onnx_op: OnnxOperation,
        graphbook_inputs: List[graphbook.Variable],
        graphbook_outputs: List[graphbook.Variable]) -> graphbook.Operation:

    """ Convert ONNX operation to Graphbook operation. """

    # Diverge here for special functions.
    if onnx_op.opType in OP_MAPPING:
        return OP_MAPPING[onnx_op.opType].convert(onnx_op, graphbook_inputs, graphbook_outputs)

    primitive_name = onnx_op.opType

    if primitive_name in OP_GB_MAPPING:
        gb_map = OP_GB_MAPPING[onnx_op.opType]
        # gb_op = GB_OPS[gb_map["op_name_map"]["gb_name"]]
        gb_op = _copy_primitive(gb_map["op_name_map"]["gb_name"])
        gb_op = _convert_from_primitive(graphbook_inputs, graphbook_outputs, gb_op, gb_map)
        gb_op.name = onnx_op.name
        return gb_op

    # There's no special function or prescribed mapping so just create blank composite
    return graphbook.Operation(
        name=onnx_op.name,
        primitive_name=str(onnx_op.opType),
        assertions=[],
        type=graphbook.OperationType.COMPOSITE_OPERATION,
        inputs=graphbook_inputs,
        outputs=graphbook_outputs,
        operations=[]
    )


