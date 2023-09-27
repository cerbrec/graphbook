from typing import List, Dict

from src import graph_util


def populate_links(
        adj_matrix: List[List[int]],
        index_to_op: Dict[int, graph_util.Operation],
        index_to_var_index: Dict[int, int]) -> List[graph_util.Link]:

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

                if source_op.primitive_name == "top":
                    link = graph_util.Link(
                        source=graph_util.LinkEndpoint(operation="this",
                                                       data=source_op.inputs[source_var_index].name),
                        sink=graph_util.LinkEndpoint(operation=sink_op.name, data=sink_op.inputs[sink_var_index].name)
                    )
                elif sink_op.primitive_name == "top":
                    link = graph_util.Link(
                        source=graph_util.LinkEndpoint(operation=source_op.name,
                                                       data=source_op.outputs[source_var_index].name),
                        sink=graph_util.LinkEndpoint(operation="this", data=sink_op.outputs[sink_var_index].name)
                    )
                else:
                    # Create link
                    link = graph_util.Link(
                        source=graph_util.LinkEndpoint(operation=source_op.name,
                                                       data=source_op.outputs[source_var_index].name),
                        sink=graph_util.LinkEndpoint(operation=sink_op.name, data=sink_op.inputs[sink_var_index].name)
                    )
                links.append(link)

    return links