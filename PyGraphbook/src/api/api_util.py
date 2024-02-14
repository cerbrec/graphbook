import requests
import json
from pydantic import BaseModel, Field, validator
from typing import List

TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IkV5Q0RTUDdHelBJZ3Q1Sl90UW52RCJ9.eyJuaWNrbmFtZSI6ImR3aW5lcjEzMSIsIm5hbWUiOiJkd2luZXIxMzFAZ21haWwuY29tIiwicGljdHVyZSI6Imh0dHBzOi8vcy5ncmF2YXRhci5jb20vYXZhdGFyL2NkM2EyZGYwNGE3NDQ4ZWFjNTk2ZWJlZGVhNjFkMmRiP3M9NDgwJnI9cGcmZD1odHRwcyUzQSUyRiUyRmNkbi5hdXRoMC5jb20lMkZhdmF0YXJzJTJGZHcucG5nIiwidXBkYXRlZF9hdCI6IjIwMjQtMDEtMDJUMDM6MjA6MzAuMzkxWiIsImVtYWlsIjoiZHdpbmVyMTMxQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJpc3MiOiJodHRwczovL2xvZ2luLmNlcmJyZWMuY29tLyIsImF1ZCI6IjhveEsxakljMTQ1MjNweDlpTzlFVjZBeGpjSzZTYnEyIiwiaWF0IjoxNzA0MTY1NjYyLCJleHAiOjE3MTQxNjU2NjIsInN1YiI6ImF1dGgwfDYxNzZlNmNkZmUzOWJiMDA2OTI5NzUzNyIsImF1dGhfdGltZSI6MTcwNDE2NTYzMCwic2lkIjoicFozd2I0Mi1URnN4Slltd2xRZW1XdVpmaElwd3hjMDQifQ.pvU9_cHmHEinXBiFrR-dkLXdG1fbwrmnpsCoy63m29SUv-KkTlti93DQumQr9E_MvPuw03POG9OpkuGtdYNju2GEDP5VWlEkfiQvMX7qmp6Dv3Sy5kvDghx-67xGNVb477HJ0gf_i-puspA_FjYFtjUdEUPQ_UZXfOoEAO0fzQYMcgjgRRuq1_oEXy8JwVyZL8mbxtWBx4BbGYC_h8dqXS8tqVEFByPxe5IiMCf6Eg-82dONcbAgR5ASTXcIpN1_kakLQt9bksfQet0eo_hqcZUjA-DLFSMIJMUE-Nbdhj5UvD5dYP8BDt8AMONlgwjlCFnfy3VKAXnzoK4myOhY7w"

"""
 {
            "userId": "10000000-0000-0000-0000-0000000000",
            "name": "weight",
            "sourceType": "ComputeResult",
            "fullPath": "/public/llama_2_chat_7b/model/layers/18/mlp/down_proj/weight",
            "referenceId": "7d30717c-3ff8-42d5-9278-bfe29d224bbe",
            "createdAt": "2024-01-06T21:50:58Z",
            "updatedAt": "2024-01-06T21:50:58Z",
            "schema": {
                "fieldId": "f8578f55-2537-4f21-8069-202d4fe6279b",
                "length": 4096,
                "elementType": {
                    "fieldId": "51ac6a01-1544-4046-b734-8e6cfcfbe498",
                    "type": "DOUBLE"
                },
                "type": "LIST"
            },
            "shape": [
                11008,
                4096
            ],
            "dataSetSize": 11008,
            "deleted": false
        },
"""


class Dataset(BaseModel):

    """ Dataset Object"""
    userId: str = Field(..., description="User ID")
    name: str = Field(..., description="Name of the dataset")
    sourceType: str = Field(..., description="Source Type")
    fullPath: str = Field(..., description="Full Path")
    referenceId: str = Field(..., description="Reference ID")
    createdAt: str = Field(..., description="Created At")
    updatedAt: str = Field(..., description="Updated At")
    extraction_schema: object = Field(..., alias="schema", description="Schema")
    shape: List[int] = Field(..., description="Shape")
    dataSetSize: int = Field(..., description="Dataset Size")
    deleted: bool = Field(..., description="Deleted")


class DatasetListResponse(BaseModel):
    """ List of Datasets in Response """
    status: str = Field(..., description="Status")
    data: List[Dataset] = Field(..., description="List of Datasets")
    meta: object = Field(..., description="Meta")


url = "https://apigw.graphbook.cerbrec.com/api/v1/datasets"


def get_dataset_list():
    payload = {}
    headers = {
      'Content-Type': 'application/json',
      'Authorization': f'Bearer {TOKEN}'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    return DatasetListResponse(**json.loads(response.text))


def get_all_full_paths() -> set:
    dataset_list = get_dataset_list()

    return {dataset.fullPath for dataset in dataset_list.data}


def get_all_redundant_full_paths() -> set:
    dataset_list = get_dataset_list()

    # Get all full paths that are there twice or more
    full_paths = [dataset.fullPath for dataset in dataset_list.data]
    return {x for x in full_paths if full_paths.count(x) > 1}



def generate_token():
    response = requests.get("https://login.cerbrec.com/authorize?client_id=8oxK1jIc14523px9iO9EV6AxjcK6Sbq2&response_type=code&code_challenge_method=S256&code_challenge=Nvb1Vwwx-UddLhsTnOnGOPCtw0_PfgA3avfV4Ufv4ws&scope=openid profile email&state=fkaymTKD&audience=https://api.cerbrec.com/&redirect_uri=http://localhost:61599/callback&prompt=login")


if __name__ == "__main__":
    generate_token()
    # print(get_all_redundant_full_paths())