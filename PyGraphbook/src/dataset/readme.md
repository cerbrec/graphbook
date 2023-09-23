# Module for constructing a graph dataset from Graphbook's graphs

## Description

This module provides a way to construct a graph dataset from Graphbook's graphs. 
This dataset can be used for learning a representation of Graphbook graphs, such as using a graph neural network, 
that can be used for tasks like making recommendations to Graphbook users as they are working,
or more generally to understand deep learning-based node graph architectures.


## Definitions

Graphbook graphs are hierarchical. Some operations are `primitive` because they have no sub-graph. 
Others are `composite` because they decompose into a sub-graph. 
Each level of the hierarchy is called a `graph level`.

The `variables` or "nodes" in the graph are the `input` and `output` variables of each operation. 
A `link` is a dependency on the data flow from an output of one operation to an input of another.
This is a [node-graph architecture](https://en.wikipedia.org/wiki/Node_graph_architecture). 

Graphbook graphs are hierarchical node-graph architectures, where the input of a composite operation flows into 
its sub-graph through a dummy operation called `this` whose outputs are the parent's inputs. The sub-graph's outputs 
flow up the parent's corresponding outputs.

![Softmax example](https://www.cerbrec.com/documentation/assets/images/screenshots/softmax_composite.png)
For example, in the sub-graph of Softmax above, the inputs on the left side correspond to the Softmax operation's inputs 
at the parent graph level, and similarly for the outputs.

Another kind of operation is considered `conditional` because it has two sub-graphs and a 
special input to determine which sub-graph to flow through.
One sub-graph is for the `true` case and the other is for the `false` case.

## Dataset Format

### Input Ids
The `Input IDs` is a 2D array of integers corresponding to vocbulary IDs of variables. Each row corresponds to a graph level.


#### Vocabulary
The vocabulary is created based on the input and output variables of each primitive.

Special vocabulary `static tensors` are also added to the vocabulary to represent when data 
is supplied with static data (i.e., bootstrapped data) rather than with a link. 
They are unique based on the data type and number of dimensions in the tensor.

Currently there are about 254 Ids in the vocabulary, where 12 of them come from static tensors.

#### Placeholder Vocabulary

The placeholder vocabulary is a subset of the vocabulary that is used to 
represent the input and output variables of each composite operation. 

```python
COMPOSITE_INPUT_ID_OFFSET = -10
COMPOSITE_OUTPUT_ID_OFFSET = -50

CONDITIONAL_INPUT_ID_OFFSET = -1000
CONDITIONAL_OUTPUT_ID_OFFSET = -5000

SUB_GRAPH_INPUT_ID_OFFSET = -100
SUB_GRAPH_OUTPUT_ID_OFFSET = -500
```

These offsets are used to indicate placeholder Ids for variables. For example, IDs `-10` and `-11` 
correspond to the first and second input of a composite operation. 
A `-1000` indicates the first input of a conditional operation.
A `-502` indicates a third output of a sub-graph (which flows to the parents third output).


### Adjacency Matrix
The `Adjacency Matrix` is a 2D array of size `n x n` where `n` is the number of nodes in the graph. 
An output variable in the `i`th position of the input Ids that supplies an input variable in the `j`th position of the input Ids 
through a directed edge in the graph is considered a `1` in 
the adjacency matrix at row `i` and column `j`. Otherwise, the adjacency matrix is `0` at that position.

### Graph Level Ids

The `Graph Level Ids` is a 2D array of integers matching the shape of the Input Ids. 

Each variable is either primitive and assigned `-1` to indicate it has no sub-graph, is a sub-graph input/output and assigned `-2`,
or it is composite and assigned a positive integer to indicate which row the variable corresponds to. 
For example, if row 0 has graph level ids = `[-2, -2, -1, 6, 6, -2]`, then the first two variables are sub-graph inputs,
the third variable is primitive, the fourth and fifth variables are inputs/outputs of composite decomposing into row 6,
and the last variable is a sub-graph output.

## Setup

Tested with: `Python 3.9.6`, `Python 3.10.5`

```bash
pip install -r requirements.txt
```

Set python path to be root folder
    
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Usage

To construct the dataset, run the following command:
```python
python3 construct_dataset.py
```

The dataset will be saved in the `graphbook_dataset` folder.

You can reconstruct a graph from the dataset using the following command:
```python
python3 reconstruct_graph.py
```