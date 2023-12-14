""" Graphbook graph object. """

from enum import Enum
from typing import List, Optional, TypeVar
from pydantic import BaseModel, Field, validator
from collections import deque


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


class Variable(BaseModel):
    """ Variable object. """

    name: str = Field(..., description="Name of the variable.")
    primitive_name: str = Field(None, description="Original name of the variable.")
    type: DataType = Field(None, description="Data Type of the variable.")
    shape: Optional[List[int]] = Field(None, description="Description of the variable.")
    onnx_attribute: Optional[bool] = Field(None, description="Whether the variable is supplied for onnx attribute")
    data: Optional[object] = Field(None, description="Bootstrapped data of the variable.")


VariableModel = TypeVar("VariableModel", str, Variable)


class LinkEndpoint(BaseModel):
    """ Link endpoint object. """

    operation: str = Field(..., description="Name of the operation.")
    data: str = Field(..., description="Name of the variable.")

    # Make hashable
    def __hash__(self):
        return hash((self.operation, self.data))


class Link(BaseModel):
    """ Link object. """

    source: LinkEndpoint = Field(..., description="Source of the link.")
    sink: LinkEndpoint = Field(..., description="Target of the link.")

    # Make hashable
    def __hash__(self):
        return hash((self.source, self.sink))


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
    inputs: List[VariableModel] = Field(..., description="List of inputs of the operation.")

    outputs: List[VariableModel] = Field(..., description="List of outputs of the operation.")

    assertions: List[str] = Field(None, description="List of assertions of the operation.")
    description: List[str] = Field(None, description="Description of the operation.")

    # Currently Disabling examples.
    # examples: List[str] = Field(..., description="Examples of the operation.")

    operations: Optional[List['Operation']] = Field(None, description="List of sub-operations of the operation.")
    links: List[Link] = Field(None, description="List of links of the operation.")

    # For conditional operations
    condition: Condition = Field(None, description="Condition of the operation.")

    operations_if_true: List['Operation'] = Field(None, description="List of sub-operations of the operation.")
    operations_if_false: List['Operation'] = Field(None, description="List of sub-operations of the operation.")

    links_if_true: List[Link] = Field(None, description="List of links of the operation.")
    links_if_false: List[Link] = Field(None, description="List of links of the operation.")

    # For Loops
    repeat_until_false_condition: RepeatUntilFalseCondition = Field(None, description="Repeat until false condition of the operation.")

    @validator('inputs', 'outputs', each_item=True)
    def convert_str_to_variable(cls, v):
        if isinstance(v, str):
            return Variable(name=v, primitive_name=v)
        return v

    # Hashable
    def __hash__(self):
        return hash(self.name)


class TopoSortMixin:
    """ Topological sort mixin. """

    def __init__(self, root: Operation):
        self.graph = root
        self.temp = set()
        self.perm = set()
        self.order = []

        self.operations = deque()
        for op in self.graph.operations:
            self.operations.append(op)

        self.adjacent_ops = {}

    def run(self):
        """ Run the topological sort. """
        while len(self.operations) > 0:
            n = self.operations.pop()
            self._visit(n)

        # Reverse sort self order
        self.order.reverse()
        self.graph.operations = self.order

    def _find_op(self, name):
        for op in self.graph.operations:
            if op.name == name:
                return op
        return None

    def _populate_adjacent_ops(self, n):

        self.adjacent_ops[n.name] = set()

        for link in self.graph.links:
            if link.source.operation == n.name:
                if link.sink.operation == "this":
                    continue

                self.adjacent_ops[n.name].add(link.sink.operation)

    def _visit(self, n):
        if n is None:
            return

        if n in self.perm:
            return

        if n in self.temp:
            raise ValueError("Not a DAG.")

        self.temp.add(n)
        if n.name not in self.adjacent_ops:
            self._populate_adjacent_ops(n)

        for op in self.adjacent_ops[n.name]:
            self._visit(self._find_op(op))

        self.temp.remove(n)
        self.perm.add(n)

        if n.operations is not None and len(n.operations) > 1:
            TopoSortMixin(n).run()

        self.order.append(n)