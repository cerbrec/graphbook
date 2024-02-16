import json
import os

path = "/Users/drw/downloads/test_img"

with open(path, 'r') as file:
    data = json.load(file)
    print("OKay")
