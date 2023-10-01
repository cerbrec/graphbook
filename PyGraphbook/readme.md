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
Populate datasets:

```bash
python3 src/dataset/hierarchical_dataset.py

python3 src/dataset/flat_dataset.py
```

Read and write specific files:

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


## Tests

Tests require you populate the datasets first.

```bash
pytest tests
```