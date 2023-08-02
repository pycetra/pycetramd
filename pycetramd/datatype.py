import json
import pathlib

current_dir = pathlib.Path(__file__).parent


def getDtype(dataset_name):
    with open(str(current_dir) + "/datatype.json", "r") as f:
        dtype_json = json.load(f)
    return list(
        zip(
            dtype_json[dataset_name]["columns"],
            dtype_json[dataset_name]["cell_type"],
        )
    )


def getDefaultValue(dataset_name):
    with open(str(current_dir) + "/datatype.json", "r") as f:
        dtype_json = json.load(f)
    return dtype_json[dataset_name]["default_dict"]
