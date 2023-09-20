# PyGraphbook

## Description

This is a collection of utility methods for working with Graphbook graphs. WIP.

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


## Tests

```bash
pytest tests
```