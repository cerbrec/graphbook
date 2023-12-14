# PyGraphbook

## Description

Graphbook represents graphs as JSON objects. 
This module provides a way to validate and work with Graphbook graphs outside the client. 

[Work in Progress]

## Setup

Currently working with `Python 3.10.5`

```bash
pip install -r requirements.txt
```

Set python path to be root folder
    
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Usage

```python
from src import graph
import json

file = "../compute_operations/math_operations/add.json"

with open(file, "r") as f:
    graph_string = f.read()
    
graph_json = json.loads(graph_string)
graph_obj = graph.Operation.model_validate(graph_json)
print(graph_obj.model_dump_json(exclude_none=True))
```

## HuggingFace to Graphbook

This module uses ONNX to convert to Graphbook.  
```
pip install optimum[exporters]

optimum-cli export onnx --model bert-base-uncased bert-base-uncased.onnx
```

```python3
python3 ./src/onnx/onnx_2_graphbook.py --onnx_folder bert-base-uncased.onnx --output_folder bert-base-uncased.graphbook
```

Some models will be decomposed into 1 or more graphs. This command above 
will produce a JSON file for each graph and a folder for the weights for each separate graph.

## Tests

```bash
pytest tests
```