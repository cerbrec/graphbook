""" Graphbook graph object. """

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


class Variable(BaseModel):
    """ Variable object. """

    name: str = Field(..., description="Name of the variable.")
    primitive_name: str = Field(None, description="Original name of the variable.")
    type: DataType = Field(None, description="Data Type of the variable.")
    shape: Optional[List[int]] = Field(None, description="Description of the variable.")


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
    inputs: List[VariableModel] = Field(..., description="List of inputs of the operation.")

    outputs: List[VariableModel] = Field(..., description="List of outputs of the operation.")

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

    @validator('inputs', 'outputs', each_item=True)
    def convert_str_to_variable(self, v):
        if isinstance(v, str):
            return Variable(name=v, primitive_name=v)
        return v

