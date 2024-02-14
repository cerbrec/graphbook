import json
from json import JSONEncoder
import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def numpy_to_json(numpy_data: np.ndarray):
    return json.dumps(numpy_data, cls=NumpyArrayEncoder)  # use dump() to write array into file


def json_examples_to_prompt_str(json_example: dict) -> str:
    """
    Script that converts JSON example such as:
         {
            "inputs": [
                {
                    "name": "array",
                    "data": [
                        [
                            [
                                0,
                                1,
                                2,
                                3,
                                4,
                                5,
                                6,
                                7,
                                8
                            ]
                        ],
                        [
                            [
                                9,
                                10,
                                11,
                                12,
                                13,
                                14,
                                15,
                                16,
                                17
                            ]
                        ]
                    ],
                    "type": "INTEGER"
                },
                {
                    "name": "dimension_index",
                    "data": 2,
                    "type": "INTEGER"
                },
                {
                    "name": "step_value",
                    "data": 3,
                    "type": "INTEGER"
                }
            ],
            "outputs": [
                {
                    "name": "sampled_array",
                    "primitive_name": "sampled_array",
                    "data": [
                        [
                            [
                                0,
                                3,
                                6
                            ]
                        ],
                        [
                            [
                                9,
                                12,
                                15
                            ]
                        ]
                    ],
                    "type": "INTEGER",
                    "shape": [
                        2,
                        1,
                        3
                    ]
                }
            ]


    To str format like:

        array=[[[0,1,2,3,4,5,6,7,8]],[[9,10,11,12,13,14,15,16,17]]]
        dimension_index=2
        step_value=3
        output=[[[0,3,6]],[[9,12,15]]]
    """
    prompt_str = ""
    for input_var in json_example["inputs"]:
        prompt_str += f"{input_var['name']}={numpy_to_json(input_var['data'])}\n"
    for output_var in json_example["outputs"]:
        prompt_str += f"{output_var['name']}={numpy_to_json(output_var['data'])}\n"
    return prompt_str


def get_examples_for_op(op):
    # Get Examples from specific operation in compute_operations/sub_dir/operation.json
    with open(f"../compute_operations/{op}.json", 'r') as file:
        operation_json = json.load(file)
        if "examples" in operation_json:
            for example in operation_json["examples"]:
                print(json_examples_to_prompt_str(example))


if __name__ == "__main__":
    get_examples_for_op("shaping_operations/expand_one_dimension")



