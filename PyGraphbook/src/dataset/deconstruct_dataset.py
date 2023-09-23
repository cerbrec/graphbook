""" Take a dataset for a graph, and a vocab set, and deconstruct it back into graph form. """

from typing import List, Optional

from src import graph_util
from src.dataset import construct_dataset


DATA_TYPE_KEYWORD = "DataType"
THIS = "this"


def _handle_last_tensor_bootstrapped(variable: graph_util.Variable, last_tensor: tuple):
    """ Handle bootstrapped data. """
    variable.type = last_tensor[0]

    num_dims = int(last_tensor[2])
    # Create nested list which is nested "num_dims" times. Only will be 0 (scalar), 1, or 2.
    if num_dims == 0:
        variable.shape = []
    elif num_dims == 1:
        variable.shape = [None]
    else:
        variable.shape = [[None]]


def _deconstruct_dataset(
    top_op_name: str,
    type_: graph_util.OperationType,
    dataset: construct_dataset.HierarchicalDataset,
    var_row: List[int],
    graph_level: List[int],
    adj_matrix: List[List[int]],
    vocab_: dict,
    level_to_op: dict,
    current_level: int = 0,
    if_true: Optional[bool] = None,
) -> graph_util.Operation:

    """ Deconstruct a dataset into a graph, hierarchically"""

    # This is a new op
    top_op = graph_util.Operation(name=top_op_name, primitive_name=top_op_name, type=type_)
    level_to_op[current_level] = top_op
    if type_ == graph_util.OperationType.CONDITIONAL_OPERATION:
        top_op.operations_if_true = []
        top_op.operations_if_false = []

    # This is a mapping from index to op.
    index_to_op = {}
    index_to_var_index = {}

    # Ops that will get assigned to this graph level.
    ops = []

    # This is tracking as per var.
    op_name_list = []

    # There is logic for deciding if each variable is a new operation or not.
    last_op_name = None
    last_op = None
    last_was_input = True

    # This is related to logic for whether or not the input is bootstrapped with shape/type.
    last_tensor = None

    # Tracking the operation names per graph level so that there are no duplicates.
    names = set()

    sub_graph_dummies = graph_util.Operation(
        name=THIS,
        primitive_name=THIS,
        type=graph_util.OperationType.PRIMITIVE_OPERATION)

    for i, (var_item, level_item) in enumerate(zip(var_row, graph_level)):
        op_name, is_input, var_name = vocab_[var_item]

        if level_item == construct_dataset.DUMMY_LEVEL:
            # Then we're in sub-graph dummy and don't need to do anything
            index_to_op[i] = sub_graph_dummies
            index_to_var_index[i] = int(var_name)
            if is_input:
                sub_graph_dummies.inputs.append(graph_util.Variable(name=f"var_{i}", primitive_name=f"var_{i}"))
            else:
                sub_graph_dummies.outputs.append(graph_util.Variable(name=f"var_{i}", primitive_name=f"var_{i}"))
            continue

        if level_item == construct_dataset.PRIMITIVE_LEVEL:

            # op_name, is_input, var_name = vocab_[var_item]
            if op_name.startswith(DATA_TYPE_KEYWORD):
                # Then this is a bootstrapped data and we'll save it for later when we do adj matrix.
                # TODO: store details of tensor.
                op_name_list.append(construct_dataset.BOOTSTRAPPED_DATA_OP_NAME)
                last_tensor = (op_name, is_input, var_name)
                index_to_op[i] = None
                index_to_var_index[i] = None
                continue

            variable = graph_util.Variable(name=var_name, primitive_name=var_name)

            if last_tensor:
                _handle_last_tensor_bootstrapped(variable, last_tensor)

            # Then it's primitive and we can add it to the graph.
            if last_op_name and last_op_name == op_name and (is_input == last_was_input or last_was_input):
                # TODO:
                """
                # There will need to be an exception here for consecutive write_to_database ops...
                # or other primitive ops that have only inputs.
                # We could handle these by having a special var or level that means "end" of operation.
                """

                # Then it's continuation of same primitive operation.
                op = last_op
            else:
                # This is a new operation.
                name = op_name
                if op_name in names:
                    # Then we've already seen this op name, so we need to create a new one.
                    name = f"{op_name}_{i}"

                op = graph_util.Operation(name=name, primitive_name=op_name,
                                          type=graph_util.OperationType.PRIMITIVE_OPERATION)
                ops.append(op)

            names.add(op.name)
            last_op_name = op_name

            if is_input:
                index_to_var_index[i] = len(op.inputs)
                op.inputs.append(variable)
            else:
                index_to_var_index[i] = len(op.outputs)
                op.outputs.append(variable)

            last_was_input = is_input

        elif var_name == construct_dataset.CONDITIONAL:
            _type = graph_util.OperationType.CONDITIONAL_OPERATION
            variable = graph_util.Variable(name=f"var_{i}", primitive_name=f"var_{i}")

            if last_tensor:
                _handle_last_tensor_bootstrapped(variable, last_tensor)

            if level_item in level_to_op:
                op = level_to_op[level_item]
            else:
                op = _deconstruct_dataset(
                    top_op_name=f"{construct_dataset.CONDITIONAL}_{i}_{current_level}",
                    type_=_type,
                    dataset=dataset,
                    var_row=dataset.variables[level_item],
                    graph_level=dataset.graph_level_ids[level_item],
                    adj_matrix=dataset.adj_matrix[level_item],
                    vocab_=vocab_,
                    level_to_op=level_to_op,
                    current_level=level_item,
                    if_true=is_input
                )

                if not is_input:
                    # Then this is the same op as last op
                    same_op = ops[-1]
                    level_to_op[level_item] = same_op
                    same_op.operations_if_false = op.operations_if_false
                    same_op.links_if_false = op.links_if_false
                    op = same_op
                else:
                    ops.append(op)

            if is_input:
                index_to_var_index[i] = len(op.inputs)
                op.inputs.append(variable)
            else:
                index_to_var_index[i] = len(op.outputs)
                op.outputs.append(variable)

            last_was_input = True
            last_op_name = op.name

        else:

            variable = graph_util.Variable(name=f"var_{i}", primitive_name=f"var_{i}")

            if last_tensor:
                _handle_last_tensor_bootstrapped(variable, last_tensor)

            # Then it's referring to a new graph level. We should recursively call this function.
            if level_item in level_to_op:
                # Then we've seen this op, so just add input/output
                op = level_to_op[level_item]

            else:
                _type = graph_util.OperationType.COMPOSITE_OPERATION
                sub_op_name = f"{construct_dataset.COMPOSITE}_{i}_{current_level}"

                op = _deconstruct_dataset(
                    top_op_name=sub_op_name,
                    type_=_type,
                    dataset=dataset,
                    var_row=dataset.variables[level_item],
                    graph_level=dataset.graph_level_ids[level_item],
                    adj_matrix=dataset.adj_matrix[level_item],
                    vocab_=vocab_,
                    level_to_op=level_to_op,
                    current_level=level_item,
                    if_true=None
                )

                ops.append(op)

            if is_input:
                index_to_var_index[i] = len(op.inputs)
                op.inputs.append(variable)
            else:
                index_to_var_index[i] = len(op.outputs)
                op.outputs.append(variable)

            last_was_input = True
            last_op_name = op.name

        index_to_op[i] = op
        last_op = op
        op_name_list.append(op.name)
        last_tensor = None


    links = []
    # Now handle the adj matrix
    for i, row in enumerate(adj_matrix):
        for j, value in enumerate(row):
            if value == 1:
                # Then there is a link from the operation output represented by the
                # row index to the operation input represented by the column index.
                source_op = index_to_op[i]
                source_var_index = index_to_var_index[i]
                sink_op = index_to_op[j]
                sink_var_index = index_to_var_index[j]
                if not source_op or not sink_op:
                    # Then this is a bootstrapped data and we don't need to do anything.
                    continue

                # Create link
                link = graph_util.Link(
                    source=graph_util.LinkEndpoint(operation=source_op.name, data=source_op.outputs[source_var_index].name),
                    sink=graph_util.LinkEndpoint(operation=sink_op.name, data=sink_op.inputs[sink_var_index].name)
                )
                links.append(link)

    if if_true:
        top_op.operations_if_true = ops
        top_op.links_if_true = links
    elif not if_true and type_ == graph_util.OperationType.CONDITIONAL_OPERATION:
        top_op.operations_if_false = ops
        top_op.links_if_false = links
    else:
        top_op.operations = ops
        top_op.links = links
    return top_op

def deconstruct_dataset(
    dataset: construct_dataset.HierarchicalDataset,
    vocab_: dict
) -> graph_util.Operation:
    """ Deconstruct a dataset into a graph. """

    var_row = dataset.variables[0]
    graph_level = dataset.graph_level_ids[0]
    adj_matrix = dataset.adj_matrix[0]

    operation = _deconstruct_dataset(
        top_op_name=dataset.name,
        type_=graph_util.OperationType.COMPOSITE_OPERATION,
        dataset=dataset,
        var_row=var_row,
        graph_level=graph_level,
        adj_matrix=adj_matrix,
        vocab_=vocab_,
        level_to_op={},
        if_true=None
    )

    if dataset.if_false_subgraph_level:
        # Then we have a conditional operation.
        operation.type = graph_util.OperationType.CONDITIONAL_OPERATION
        if_false_op = _deconstruct_dataset(
            top_op_name=dataset.name,
            type_=graph_util.OperationType.COMPOSITE_OPERATION,
            dataset=dataset,
            var_row=dataset.variables[dataset.if_false_subgraph_level],
            graph_level=dataset.graph_level_ids[dataset.if_false_subgraph_level],
            adj_matrix=dataset.adj_matrix[dataset.if_false_subgraph_level],
            vocab_=vocab_,
            level_to_op={},
            if_true=None
        ).operations

        operation.operations_if_false = if_false_op.operations
        operation.links_if_false = if_false_op.links
        operation.operations_if_true = operation.operations
        operation.links_if_true = operation.links
        operation.operations = None
        operation.links = None

    return operation

