import json
import logging

from src import graph_util
from src.dataset import dataset_util
from src.dataset import flat_dataset


def reconstruct_dataset(
        dataset_obj: flat_dataset.Dataset,
        vocab: dict) -> graph_util.Operation:
    """ Reconstruct the dataset. """

    top_op = graph_util.Operation(
        name=dataset_obj.name,
        primitive_name=flat_dataset.TOP,
        type=graph_util.OperationType.COMPOSITE_OPERATION)

    top_op.operations = []

    index_to_op = {}
    index_to_var_index = {}
    last_op = None
    last_operation_name = ""
    last_was_input = False
    names_so_far = set()
    for i, var_id in enumerate(dataset_obj.variables):
        operation_name, is_input, var_name = vocab[var_id]

        if operation_name == flat_dataset.SPECIAL_OPERATION_NAMES[0]:
            # Then it's conditional
            # We necessarily have to build in

            if operation_name == last_operation_name and (is_input == last_was_input or last_was_input):
                # Then this is continuation.
                op = top_op
            else:
                original_name = operation_name
                if operation_name in names_so_far:
                    operation_name = f"{operation_name}_{i}"

                names_so_far.add(operation_name)

                op = graph_util.Operation(
                    name=operation_name,
                    primitive_name=original_name,
                    type=graph_util.OperationType.CONDITIONAL_OPERATION)

                top_op.operations.append(op)
                # But how do we know which operations belong in here, vs outside of here? for now, don't worry about it?

            pass
        elif operation_name == flat_dataset.SPECIAL_OPERATION_NAMES[1]:

            if var_name == "Looping Data Parent Input" or var_name == "Looping Data Sub-graph Input":
                # Then it's the parent intro to the loop body:

                # Then it's loop body
                if is_input == last_was_input or last_was_input:
                    # Then this is continuation.
                    op = top_op
                else:
                    original_name = operation_name
                    if operation_name in names_so_far:
                        operation_name = f"{operation_name}_{i}"

                    names_so_far.add(operation_name)

                    op = graph_util.Operation(
                        name=operation_name,
                        primitive_name=original_name,
                        type=graph_util.OperationType.LOOP_BODY_OPERATION)

                    top_op.operations.append(op)
            else:
                # Then it's loop data output.
                if is_input == last_was_input or last_was_input:
                    # Then this is continuation.
                    op = top_op
                else:
                    original_name = operation_name
                    if operation_name in names_so_far:
                        operation_name = f"{operation_name}_{i}_run_again"

                    names_so_far.add(operation_name)

                    op = graph_util.Operation(
                        name=operation_name,
                        primitive_name=original_name,
                        type=graph_util.OperationType.PRIMITIVE_OPERATION)

                    top_op.operations.append(op)

        # var_name == "Looping Data Parent Input"
        # vocab.append((len(vocab), (SPECIAL_OPERATION_NAMES[1], True, "Run Again")))
        # vocab.append((len(vocab), (SPECIAL_OPERATION_NAMES[1], True, "Looping Data Sub-graph Input")))

        elif operation_name == flat_dataset.TOP:
            # Then it's top level
            op = top_op
        # Is it a new operation?
        elif operation_name == last_operation_name and (is_input == last_was_input or last_was_input):
            # Is same operation.
            op = last_op
        else:
            # Is new operation.
            original_name = operation_name
            if operation_name in names_so_far:
                operation_name = f"{operation_name}_{i}"

            names_so_far.add(operation_name)

            op = graph_util.Operation(
                name=operation_name,
                primitive_name=original_name,
                type=graph_util.OperationType.PRIMITIVE_OPERATION)

            top_op.operations.append(op)

        variable = graph_util.Variable(
            name=var_name,
            primitive_name=var_name)

        if is_input:
            index_to_var_index[i] = len(op.inputs)
            op.inputs.append(variable)
        else:
            index_to_var_index[i] = len(op.outputs)
            op.outputs.append(variable)

        # What about links?

        index_to_op[i] = op
        last_op = op
        last_operation_name = op.primitive_name
        last_was_input = is_input

    links = dataset_util.populate_links(dataset_obj.adj_matrix, index_to_op, index_to_var_index)
    top_op.links = links

    return top_op


if __name__ == "__main__":

    # file_name = "flat_dataset/graphs/Softmax.json"
    # file_name = "flat_dataset/graphs/Adam_Optimizer.json"
    file_name = "flat_dataset/graphs/Add_Weight.json"
    # file_name = "flat_dataset/graphs/GPT-2_+_Next_Token_Head.json"

    with open(file_name, "r") as f:
        dataset_json = json.load(f)
        dataset = flat_dataset.Dataset.model_validate(dataset_json)

    logging.info("Deserialized dataset")

    with open("flat_dataset/vocab.json", "r") as f:
        _vocab = json.load(f)

    _vocab = {vocab_level[0]: vocab_level[1] for vocab_level in _vocab}

    op = reconstruct_dataset(dataset, _vocab)
    print(op)
