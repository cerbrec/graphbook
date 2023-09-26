""" Graphbook graph object. """

import json
from enum import Enum
from typing import List, Optional, TypeVar
from pydantic import BaseModel, Field, validator


class OperationType(str, Enum):
    """ Operation type. """

    PROJECT = "PROJECT"
    PRIMITIVE_OPERATION = "PRIMITIVE_OPERATION"
    COMPOSITE_OPERATION = "COMPOSITE_OPERATION"
    CONDITIONAL_OPERATION = "CONDITIONAL_OPERATION"
    LOOP_OPERATION = "LOOP_OPERATION"
    LOOP_BODY_OPERATION = "LOOP_BODY_OPERATION"
    LOOP_INIT_OPERATION = "LOOP_INIT_OPERATION"


class DataType(str, Enum):
    """ Data Type """

    DECIMAL = "DECIMAL"
    INTEGER = "INTEGER"
    TEXT = "TEXT"
    BOOLEAN = "BOOLEAN"
    INTEGER_OR_DECIMAL = "INTEGER_OR_DECIMAL"
    NULL = "NULL"


class FlowState(str, Enum):
    """ Flow State """

    BOOT_SOURCE = "BOOT_SOURCE"
    REF_SOURCE = "REF_SOURCE"
    BOOT_SINK = "BOOT_SINK"
    REF_SINK = "REF_SINK"
    LOOP_PARAMETER = "LOOP_PARAMETER"
    UNBOUND = "UNBOUND"


class Variable(BaseModel):
    """ Variable object. """

    name: str = Field(..., description="Name of the variable.")
    primitive_name: str = Field(None, description="Original name of the variable.")
    type: DataType = Field(None, description="Data Type of the variable.")
    shape: Optional[List[int]] = Field(None, description="Description of the variable.")
    flow_state: Optional[FlowState] = Field(None, description="Flow state of the variable.")
    global_constant: Optional[str] = Field(None, description="Global constant of the variable.")


VariableModel = TypeVar("VariableModel", str, Variable)


class LinkEndpoint(BaseModel):
    """ Link endpoint object. """

    operation: str = Field(..., description="Name of the operation.")
    data: str = Field(..., description="Name of the variable.")


class Link(BaseModel):
    """ Link object. """

    source: LinkEndpoint = Field(..., description="Source of the link.")
    sink: LinkEndpoint = Field(..., description="Target of the link.")


class Condition(BaseModel):
    name: str = Field(..., description="Name of the condition.")


class RepeatUntilFalseCondition(BaseModel):
    """ repeat_until_false_condition":{"name":"Run Again","loop_data":["Iteration","Next Token","Concat Tokens"]} """
    name: str = Field(..., description="Name of the condition.")
    loop_data: List[str] = Field(..., description="List of loop data.")


class Operation(BaseModel):
    """ Operation object. """

    name: str = Field(..., description="Name of the operation.")
    primitive_name: str = Field(None, description="Primitive name of the operation.")

    aliases: List[str] = Field(None, description="List of aliases of the operation.")

    # Type is required
    type: OperationType = Field(..., description="Type of the operation.")

    # Each variable can be a string or a Variable object, but they all should become Variable objects when deserialized.
    inputs: List[VariableModel] = Field(default_factory=list, description="List of inputs of the operation.")

    outputs: List[VariableModel] = Field(default_factory=list, description="List of outputs of the operation.")

    assertions: List[str] = Field(None, description="List of assertions of the operation.")
    description: List[str] = Field(None, description="Description of the operation.")

    # Currently Disabling examples.
    # examples: List[str] = Field(..., description="Examples of the operation.")

    operations: List['Operation'] = Field(None, description="List of sub-operations of the operation.")
    links: List[Link] = Field(None, description="List of links of the operation.")

    # For conditional operations
    condition: Condition = Field(None, description="Condition of the operation.")

    operations_if_true: List['Operation'] = Field(None, description="List of sub-operations of the operation.")
    operations_if_false: List['Operation'] = Field(None, description="List of sub-operations of the operation.")

    links_if_true: List[Link] = Field(None, description="List of links of the operation.")
    links_if_false: List[Link] = Field(None, description="List of links of the operation.")

    # For Loops
    repeat_until_false_condition: RepeatUntilFalseCondition = Field(None, description="Repeat until false condition of the operation.")

    # For project level
    global_constants: List[VariableModel] = Field(None, description="List of global constants of the project.")

    def __init__(self, **schema):
        """ Override init to do some error correcting.

        For extra context about the below error-correcting code:

        Within Graphbook, the actual sub-graph in Loop Operations are hidden from users. Loop operations appear to
        have two sub-graphs, a Loop Init and a Loop Body, but they are actually coming from sub-graphs of operations
        within the hidden sub-graph of Loop Operation. When you click on Loop Init or Loop Body, you go to the
        "grand-child sub-graph" of the Loop Operation. The links within this hidden sub-graph are fixed and not
        editable to the user. However... there was once a bug where the two operations (hidden from user) which
        accidentally named both sub-operations the same thing, both ending with Loop Body instead of Loop Init
        and Loop Body. As a result, the links were messed up as well. This has already been fixed, but some old
        graphbook templates may contain this bug. Thus, we correct for this by making sure that any loop init type
        operation is renamed to Loop Init, and then all links are generated from scratch (since these links are always
        the same in this hidden sub-graph). Feel free to reach out on the Slack community channel.
        """

        if schema['type'] == OperationType.LOOP_OPERATION:
            # Need to do some error correcting for the links.
            # For each sub-graph input, there should be a link to both the loop init and loop body.
            # For each loop init output, there should be a link to the loop body.
            # For each loop body output, there should be a link to the sub-graph output
            num_init_outputs_offset = len(schema["operations"][0]["outputs"])
            num_looping_data = len(schema["operations"][1]["repeat_until_false_condition"]["loop_data"])
            schema["operations"][0]["name"] = schema["operations"][0]["name"].replace("Loop Body", "Loop Init")
            schema["links"] = [
                {
                    # From init to each loop init input
                    "source": {
                        "operation": "this",
                        "data": schema["inputs"][i]["name"]
                    },
                    "sink": {
                        "operation": schema["operations"][0]["name"],
                        "data": schema["operations"][0]["inputs"][i]["name"]
                    }
                }
                for i in range(len(schema["inputs"]))
            ] + [
                # From init to each loop body input
                {
                    "source": {
                        "operation": "this",
                        "data": schema["inputs"][i]["name"]
                    },
                    "sink": {
                        "operation": schema["operations"][1]["name"],
                        # This has an offset because the loop body inputs are after the loop init inputs
                        "data": schema["operations"][1]["inputs"][num_init_outputs_offset + i]["name"]
                    }
                }
                for i in range(len(schema["inputs"]))
            ] + [
                {
                    # From each loop body output to the sub-graph output
                    "source": {
                        "operation": schema["operations"][1]["name"],
                        "data": schema["operations"][1]["outputs"][i+num_looping_data+1]["name"]
                    },
                    "sink": {
                        "operation": "this",
                        "data": schema["outputs"][i]["name"]
                    }
                }
                for i in range(len(schema["outputs"]))
            ] + [
                # From each loop init output to the loop body
                {
                    "source": {
                        "operation": schema["operations"][0]["name"],
                        "data": schema["operations"][0]["outputs"][i]["name"]
                    },
                    "sink": {
                        "operation": schema["operations"][1]["name"],
                        "data": schema["operations"][1]["inputs"][i]["name"]
                    }
                }
                for i in range(len(schema["operations"][0]["outputs"]))

            ]

        super().__init__(**schema)

    @validator('inputs', 'outputs', each_item=True)
    def convert_str_to_variable(cls, v):
        if isinstance(v, str):
            return Variable(name=v, primitive_name=v)

        if v.primitive_name is None:
            v.primitive_name = v.name

        return v


# def fetch_suppliers(
#         variable: Variable,
#         operation: Operation,
#         links: List[Link],
#         link_endpoints_map: dict,
#         vocab: dict):
#
#     suppliers = []
#     for link in links:
#         if link.sink.operation == operation.name and link.sink.data == variable.name:
#             # Then this is a match
#             operation, var = link_endpoints_map[link.sink]
#             if operation.type == OperationType.PRIMITIVE_OPERATION:
#
#


def calculate_num_graph_levels(graph: Operation) -> int:
    """ Calculate how many levels a graph has.

    A level is defined as every Operation object in the nested graph.
    """

    if graph is None:
        return 0

    if graph.type == OperationType.CONDITIONAL_OPERATION:
        # 1 for each sub-level plus the graph levels of the if true and if false sub-graphs.
        return 2 + \
            sum(calculate_num_graph_levels(operation) for operation in graph.operations_if_true) + \
            sum(calculate_num_graph_levels(operation) for operation in graph.operations_if_false)

    if graph.operations is None:
        # This is primitive, so it shouldn't count as a level.
        return 0

    # This graph level plus the graph levels of all sub-operations. If primitive, will add 0.
    return 1 + sum(calculate_num_graph_levels(operation) for operation in graph.operations)


def calculate_graph_maximum_height(graph: Operation) -> int:
    """ Calculate the maximum height of a graph. """

    if graph is None:
        return 0

    if graph.type == OperationType.CONDITIONAL_OPERATION:
        return 1 + \
            max(calculate_graph_maximum_height(operation) for operation in graph.operations_if_true) + \
            max(calculate_graph_maximum_height(operation) for operation in graph.operations_if_false)

    if graph.operations is None:
        return 0

    return 1 + max(calculate_graph_maximum_height(operation) for operation in graph.operations)


def calculate_num_links_in_graph(graph: Operation) -> int:
    """ Calculate the number of links in a graph. """

    if graph is None:
        return 0

    if graph.type == OperationType.CONDITIONAL_OPERATION:
        return len(graph.links_if_true) + len(graph.links_if_false) + \
            sum(calculate_num_links_in_graph(operation) for operation in graph.operations_if_true) + \
            sum(calculate_num_links_in_graph(operation) for operation in graph.operations_if_false)

    if graph.links is None:
        return 0

    return len(graph.links) + sum(calculate_num_links_in_graph(operation) for operation in graph.operations)


def calculate_num_primitives_in_graph(graph: Operation) -> int:
    """ Calculate the number of primitives in a graph. """

    if graph is None:
        return 0

    if graph.type == OperationType.CONDITIONAL_OPERATION:
        return sum(calculate_num_primitives_in_graph(operation) for operation in graph.operations_if_true) + \
            sum(calculate_num_primitives_in_graph(operation) for operation in graph.operations_if_false)

    if graph.operations is None:
        return 1

    return sum(calculate_num_primitives_in_graph(operation) for operation in graph.operations)


def read_graphbook_from_file(file_path: str):
    """ Read graphbook from file. """

    with open(file_path, "r") as f:
        graph_json = json.load(f)
        return Operation.model_validate(graph_json)