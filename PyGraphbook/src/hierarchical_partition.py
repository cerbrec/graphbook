import json
import logging
import uuid
from collections import defaultdict
import networkx as nx
from src import graph as graphbook

MAX_OPS_IN_GRAPH = 12


"""
Plans for hierarchical partitioning:

ALGORITHM
1. Given graph, generate all possible compositions of a new composite operation.
2. Score each candidate composition based on the following:
    a. The sum of all the link distances between the operations in the composition.
    b. The number of inputs of the new composite operation
    c. The number of outputs of the new composite operation
3. Select the composition with the lowest score
4. Repeat until the graph is partition below minimize size
5. Repeat for each sub-operation. 

METHODS
- generate_candidates()
- score_candidate()
"""

def _generate_candidates(graph: graphbook.Operation) -> list:
    """
    Generate all possible candidates for a new composite operation.
    :param graph: The graph to generate candidates for.
    :return: A list of all possible candidates.
    """
    candidates = []

    # Composite must contain at least 2 operations
    for i, op in enumerate(graph.operations[:-2]):
        for j, op2 in enumerate(graph.operations[i+2:]):

            # Composite cannot contain all operations
            if j-i == len(graph.operations):
                continue

            candidates.append((op.name, op2.name))

    return candidates




def _push_subgraph_into_composite(first_op: str, last_op: str, graph: graphbook.Operation) -> bool:

    new_composite_name = f"{first_op}_{str(uuid.uuid4())[:8]}"

    op_to_links = defaultdict(set)
    for link in graph.links:
        op_to_links[link.sink.operation].add(link)
        op_to_links[link.source.operation].add(link)

    # Determine all operations between the start and end of the composite
    composite_sub_ops = []
    start_collecting = False
    index_of_first_op = None
    for i, operation in enumerate(graph.operations):
        if operation.name == first_op:
            start_collecting = True
            composite_sub_ops.append(operation)
            index_of_first_op = i
            continue

        if operation.name == last_op:
            composite_sub_ops.append(operation)
            break

        if start_collecting:
            composite_sub_ops.append(operation)

    if index_of_first_op is None:
        raise ValueError(f"Could not find first op: {first_op} in graph: {graph.name}")

    if len(composite_sub_ops) <= 1:
        # Then canot do anything here, get next highest scoring link.
        return False

    # Now we've got all operations that need to be here. Next, determine all inputs and outputs
    # that are coming from outside the composite/ going out from the composite
    composite_inputs = []
    composite_outputs = []
    internal_links = []
    parent_links = []
    links_to_remove = set()

    composite_names = [operation.name for operation in composite_sub_ops]

    for sub_op in composite_sub_ops:

        # Since this sub_op will be removed, so will all the links to it.
        links_to_remove.update(op_to_links[sub_op.name])

        for link in op_to_links[sub_op.name]:

            if link.source.operation not in composite_names:

                # Collect the inputs that come from outside the perimeter of the composite
                if link.source.data not in composite_inputs:
                    # The sink data is the composite input.
                    composite_inputs.append(link.source.data)

                # And add internal link to "THIS"
                internal_links.append(graphbook.Link(
                    source=graphbook.LinkEndpoint(operation="this", data=link.source.data),
                    sink=graphbook.LinkEndpoint(operation=link.sink.operation, data=link.sink.data)
                ))

                # New link to the composite from parent level.
                parent_links.append(graphbook.Link(
                    source=graphbook.LinkEndpoint(operation=link.source.operation, data=link.source.data),
                    sink=graphbook.LinkEndpoint(operation=new_composite_name, data=link.source.data)
                ))

            elif link.sink.operation not in composite_names:

                # The collect outputs that leave the perimeter of the composite
                if link.sink.data not in composite_outputs:
                    composite_outputs.append(link.sink.data)

                # And add internal link to "THIS"
                internal_links.append(graphbook.Link(
                    source=graphbook.LinkEndpoint(operation=link.source.operation, data=link.source.data),
                    sink=graphbook.LinkEndpoint(operation="this", data=link.sink.data)
                ))

                # New link to the composite from parent level.
                parent_links.append(graphbook.Link(
                    source=graphbook.LinkEndpoint(operation=new_composite_name, data=link.sink.data),
                    sink=graphbook.LinkEndpoint(operation=link.sink.operation, data=link.sink.data)
                ))
            else:
                internal_links.append(link)

    composite = graphbook.Operation(
        name=new_composite_name,
        primitive_name=new_composite_name,
        type=graphbook.OperationType.COMPOSITE_OPERATION,
        inputs=composite_inputs,
        outputs=composite_outputs,
        operations=composite_sub_ops,
        links=internal_links
    )

    # Remove all the links to the subgraph
    links_copy = list(graph.links)
    graph.links = []
    for link in links_copy:
        if link not in links_to_remove:
            graph.links.append(link)

    # Add composite to parent level
    graph.operations.insert(index_of_first_op, composite)

    graph.links.extend(parent_links)

    # remove all the displaced operations.
    ops_copy = list(graph.operations)
    graph.operations = []
    for op in ops_copy:
        if op.name not in composite_names:
            graph.operations.append(op)

    return True





def _calculate_shareability_scores(graph: graphbook.Operation) -> dict:
    """
        Calculate the shareability scores for each operation in the graph.

        Shareability score is the number of link sources that are shared with predecessor operation in the graph
        minus the number of link sources that are not shared.
        If the link source includes the predecessor, then that counts for the numerator

    """

    op_to_score = defaultdict(int)
    last_op = "this"
    last_op_sources = set()
    for op in graph.operations:
        links_to_op = [link for link in graph.links if link.sink.operation == op.name]
        for link in links_to_op:
            if link.source.operation == last_op:
                op_to_score[op.name] += 1
            elif link.source in last_op_sources:
                op_to_score[op.name] += 1
            else:
                op_to_score[op.name] -= 0.5

        last_op = op.name
        last_op_sources = set([link.source.operation for link in links_to_op])

    return dict(sorted(op_to_score.items(), key=lambda item: item[1], reverse=False))




def _calculate_edge_betweenness_centrality(graph: graphbook.Operation) -> dict:
    """
    Calculate the edge betweenness centrality for each edge in the graph.
    :param graph: The graph to calculate the edge betweenness centrality for.
    :return: A map of each edge to its edge betweenness centrality.
    """
    # Create map of each operation to its links backwards
    # op_to_links = {}
    score_to_link = {}
    nx_graph = nx.Graph()
    for link in graph.links:
        if link.source.operation == "this":
            nx_graph.add_edge("Input", link.sink.operation)
        elif link.sink.operation == "this":
            nx_graph.add_edge(link.source.operation, "Output")
        else:
            nx_graph.add_edge(link.source.operation, link.sink.operation)
        # if link.sink.operation not in op_to_links:
        #     op_to_links[link.sink.operation] = []
        # op_to_links[link.sink.operation].append(link)

    for key, value in nx.edge_betweenness_centrality(nx_graph).items():
        if "Input" in key or "Output" in key:
            continue

        # Align link to be directed
        for op in graph.operations:
            if op.name == key[0]:
                score_to_link[value] = key
                break
            elif op.name == key[1]:
                score_to_link[value] = (key[1], key[0])
                break

            # score_to_link[value] = key

    return dict(sorted(score_to_link.items(), key=lambda item: item[0], reverse=True))


def _partition(op: graphbook.Operation, max_ops_in_graph: int = MAX_OPS_IN_GRAPH):
    """
    Partition the operation into sub-operations.
    :param op: The operation to partition.
    :return: The partitioned operation.
    """

    if op.operations is None:
        return

    last_len = len(op.operations)

    # If the operation has less than MAX_OPS_IN_GRAPH operations, return the operation itself.
    while len(op.operations) > max_ops_in_graph:

        logging.debug(f"Remaining {op.name}, operations: {len(op.operations)}")

        score_to_op = _calculate_shareability_scores(op)
        # score_to_link = _calculate_edge_betweenness_centrality(op)

        if len(score_to_op) == 0:
            break

        # Get top scoring link
        for top_scoring_op, score in score_to_op.items():
            if op.operations[0].name == top_scoring_op or op.operations[-1].name == top_scoring_op:
                continue

            logging.debug(f"Score: {score}, len(op.operations): {len(op.operations)}")
            try:
                success = _push_subgraph_into_composite(top_scoring_op, op.operations[-1].name, op)
            except Exception as e:
                # _calculate_edge_betweenness_centrality(op)
                success = False
            if success:
                break

        if last_len == len(op.operations):
            # Guess we couldn't do anything about it
            logging.debug(f"Could not partition {op.name}")
            break

        last_len = len(op.operations)


def recursive_partition(top_op: graphbook.Operation, max_ops_in_graph: int = MAX_OPS_IN_GRAPH):
    """ Recursively partition the operation. """

    for op in top_op.operations:
        logging.debug(f"Partitioning {op.name}")
        _partition(op, max_ops_in_graph)
        if op.operations is None or len(op.operations) == 0:
            continue
        recursive_partition(op, max_ops_in_graph)


if __name__ == "__main__":

    # Set logging mode to debug
    logging.basicConfig(level=logging.DEBUG)

    with open('./flan-t5-small-graphbook/decoder_model.onnx.json') as f:
        data = json.load(f)

    _op = graphbook.Operation(**data)

    # Sort
    graphbook.TopoSortMixin(_op).run()

    # Partition the graph
    recursive_partition(_op)

    # Save the new graph to a file
    with open("partitioned_decoder_model.onnx.json", 'w') as f:
        f.write(_op.model_dump_json(indent=4, exclude_none=True))

    # decoder_level = op.operations[0]

    # _partition(decoder_level)

