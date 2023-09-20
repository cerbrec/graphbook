# Module for constructing a graph dataset from Graphbook's graphs

## Description

This module provides a way to construct a graph dataset from Graphbook's graphs. 
This dataset can be used for learning a representation of Graphbook graphs, such as using a graph neural network, 
that can be used for tasks like making recommendations to Graphbook users as they are working,
or more generally to understand deep learning-based node graph architectures.

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
```
