""" Graphbook graph object. """

from enum import Enum
from typing import List, Optional, TypeVar, Any
from pydantic import BaseModel, Field, field_validator, validator


class OperationType(str, Enum):
    """ Operation type. """

    PRIMITIVE_OPERATION = "PRIMITIVE_OPERATION"
    COMPOSITE_OPERATION = "COMPOSITE_OPERATION"
    CONDITIONAL_OPERATION = "CONDITIONAL_OPERATION"
    LOOP_OPERATION = "LOOP_OPERATION"

from pydantic.functional_validators import WrapValidator
from typing_extensions import Annotated

class Variable(BaseModel):
    """ Variable object. """

    name: str = Field(..., description="Name of the variable.")
    primitive_name: str = Field(None, description="Original name of the variable.")
    type: str = Field(None, description="Data Type of the variable.")
    shape: Optional[List[int]] = Field(None, description="Description of the variable.")



VariableModel = TypeVar("VariableModel", str, Variable)


class Operation(BaseModel):
    """ Operation object. """

    name: str = Field(..., description="Name of the operation.")
    primitive_name: str = Field(..., description="Primitive name of the operation.")

    aliases: List[str] = Field(..., description="List of aliases of the operation.")

    # Type is required
    type: OperationType = Field(..., description="Type of the operation.")

    # Each variable can be a string or a Variable object, but they all should become Variable objects when deserialized.
    inputs: List[VariableModel] = Field(..., description="List of inputs of the operation.")

    outputs: List[VariableModel] = Field(..., description="List of outputs of the operation.")

    # Currently disabling these
    assertions: List[str] = Field(..., description="List of assertions of the operation.")
    description: List[str] = Field(..., description="Description of the operation.")
    # examples: List[str] = Field(..., description="Examples of the operation.")

    @validator('inputs', each_item=True)
    def convert_str_to_variable(cls, v):
        if isinstance(v, str):
            return Variable(name=v, primitive_name=v)
        return v

    @validator('outputs', each_item=True)
    def convert_str_to_variable(cls, v):
        if isinstance(v, str):
            return Variable(name=v, primitive_name=v)
        return v


