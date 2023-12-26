import json
import logging
import uuid
import copy
from collections import defaultdict
import networkx as nx
from typing import Tuple, List
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


def partition_algorithm(graph: graphbook.Operation, max_ops_in_graph: int = MAX_OPS_IN_GRAPH):
    """
    Partition the graph into smaller sub-graphs.
    :param max_ops_in_graph:
    :param graph: The graph to partition.
    :return: The partitioned graph.
    """
    best_graph = graph
    # last_graph = graph

    while len(best_graph.operations) >= max_ops_in_graph:

        logging.debug(f"Partitioning {graph.name}" + ("." * len(best_graph.operations)))

        # last_graph = best_graph
        best_graph = _select_min_composition(best_graph)

        if best_graph is None:
            raise ValueError("Could not partition graph.")

    sub_ops = []
    # Recursion
    for op in best_graph.operations:
        # Partitioning an operation means partitioning its sub-operations.
        if op.operations is not None and len(op.operations) > 0:
            best_sub = partition_algorithm(op, max_ops_in_graph)
            sub_ops.append(best_sub)
        else:
            sub_ops.append(op)

    best_graph.operations = sub_ops

    return best_graph


def _select_min_composition(graph: graphbook.Operation) -> graphbook.Operation:
    """
    Select the composition with the lowest score.
    :param graph: The graph to select the composition for.
    :return: The composition with the lowest score.
    """
    candidates = _generate_candidates(graph)
    if len(candidates) == 0:
        return None

    return min(candidates, key=lambda x: x[0])[1]


def _generate_candidates(graph: graphbook.Operation) -> List[Tuple[float, graphbook.Operation]]:
    """
    Generate all possible candidates for a new composite operation.
    :param graph: The graph to generate candidates for.
    :return: A list of all possible candidates.
    """
    candidates = []

    # Composite must contain at least 2 operations
    for i, op in enumerate(graph.operations[:-1]):
        for j, op2 in enumerate(graph.operations[i + 1:]):

            j_index = j + i + 1

            # Composite cannot contain all operations
            if j_index - i == len(graph.operations) - 1:
                continue

            new_graph, new_comp = _gen_candidate(graph, op, op2)
            score = _score_graph(new_graph, new_comp)
            candidates.append((score, new_graph))

    return candidates


def _score_graph(graph: graphbook.Operation, new_op: graphbook.Operation) -> float:
    """
    Score the graph based on the following:
    a. The sum of all the link distances between the operations in the composition.
    b. The number of inputs of the new composite operation
    c. The number of outputs of the new composite operation
    :param graph: The graph to score.
    :return: The score of the graph.
    """
    if graph is None:
        return float('inf')

    score = 0
    # Sum of the link distances to the new composite operation.
    for link in graph.links:
        if link.sink.operation == new_op.name:
            score += _calculate_link_distance(graph, link)

    score += len(new_op.inputs) * 10
    score += len(new_op.outputs)

    # reward for number of operations in the composite
    if len(new_op.operations) <= 10:
        # This rewards MORE operations in the composite, closer to 10
        score -= len(new_op.operations)
    elif len(new_op.operations) >= 20:
        # This penalizes TOO MANY operations in the composite, closer to 20
        score += len(new_op.operations)

    return score


def _calculate_link_distance(graph: graphbook.Operation, link: graphbook.Link) -> int:
    """
    Calculate the link distance between two operations.
    :param graph: The graph to calculate the link distance for.
    :param link: The link to calculate the link distance for.
    :return: The link distance between the two operations.
    """
    op_names = ["this"] + [op.name for op in graph.operations]

    source_index = op_names.index(link.source.operation)
    sink_index = op_names.index(link.sink.operation)

    # The distance is the space between them minus 1
    return abs(source_index - sink_index) - 1


def _gen_candidate(graph: graphbook.Operation,
                   first_op: graphbook.Operation,
                   last_op: graphbook.Operation, ) -> Tuple[graphbook.Operation, graphbook.Operation]:
    """ Generate Candidate graph where first through last op are composed into composite """

    new_composite_name = f"partition_{str(uuid.uuid4())[:4]}"

    # Make a copy of the graph
    new_graph = copy.copy(graph)
    new_graph.operations = copy.copy(graph.operations)
    new_graph.links = copy.copy(graph.links)

    # new_graph = graphbook.Operation(**graph.model_dump())

    op_to_links = defaultdict(set)
    for link in graph.links:
        op_to_links[link.sink.operation].add(link)
        op_to_links[link.source.operation].add(link)

    composite_sub_ops = []
    start_collecting = False
    index_of_first_op = None

    for i, operation in enumerate(new_graph.operations):
        if operation.name == first_op.name:
            start_collecting = True
            composite_sub_ops.append(operation)
            index_of_first_op = i
            continue

        if operation.name == last_op.name:
            composite_sub_ops.append(operation)
            break

        if start_collecting:
            composite_sub_ops.append(operation)

    if index_of_first_op is None:
        raise ValueError(f"Could not find first op: {first_op} in graph: {graph.name}")

    if len(composite_sub_ops) <= 1:
        # Then canot do anything here, get next highest scoring link.
        return None

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
    new_graph.links = []
    for link in links_copy:
        if link not in links_to_remove:
            new_graph.links.append(link)

    # Add composite to parent level
    new_graph.operations.insert(index_of_first_op, composite)

    new_graph.links.extend(parent_links)

    # remove all the displaced operations.
    ops_copy = list(new_graph.operations)
    new_graph.operations = []
    for op in ops_copy:
        if op.name not in composite_names:
            new_graph.operations.append(op)

    return new_graph, composite


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

    _op = partition_algorithm(_op)

    # Partition the graph
    # recursive_partition(_op)

    # Save the new graph to a file
    with open("partitioned_decoder_model.onnx.json", 'w') as f:
        f.write(_op.model_dump_json(indent=4, exclude_none=True))

    # decoder_level = op.operations[0]

    # _partition(decoder_level)
