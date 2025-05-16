import argparse
import json
import shutil

import pandas as pd
from datasets import Dataset


def main(ds_src: str, ds_dest: str, ds_name: str) -> None:
    splits = _create_splits(ds_src, ds_dest, ds_name)
    _create_dataset_info(ds_dest, splits)


def _create_splits(ds_src: str, ds_dest: str, ds_name: str) -> dict:
    splits = {}
    for split in ["train", "valid", "test"]:
        row_count = _process_ds(ds_src + f"/msg-{split}.jsonl", ds_dest, ds_name, split)
        splits[split] = {
            "name": split,
            "num_examples": row_count,
            "dataset_name": ds_name
        }

    return splits


def _process_ds(ds_path: str, ds_dest: str, ds_name: str, split: str) -> int:
    instructions = []
    inputs = []
    outputs = []
    df = pd.read_json(ds_path, lines=True, orient="records")
    for index, row in df.iterrows():
        instructions.append("Review the following diff hunk and provide a constructive comment.")
        inputs.append("The diff hunk is: '" + row["patch"] + "'")
        outputs.append(row["msg"])

    data = {"instruction": instructions, "text": inputs, "target": outputs}
    result = Dataset.from_dict(data)

    split_path = ds_dest + "/" + split
    result.save_to_disk(split_path)

    shutil.move(split_path + "/" + "data-00000-of-00001.arrow", f"{ds_dest}/{ds_name}-{split}.arrow")
    shutil.rmtree(split_path)
    return result.num_rows


def _create_dataset_info(ds_dest: str, splits: dict) -> None:
    dataset_features = {
        "instruction": {
            "dtype": "string",
            "_type": "Value"
        },
        "text": {
            "dtype": "string",
            "_type": "Value"
        },
        "target": {
            "dtype": "string",
            "_type": "Value"
        }
    }

    dataset_info = {
        "features": dataset_features,
        "splits": splits,
    }

    with open(ds_dest + "/dataset_info.json", 'w') as f:
        json.dump(dataset_info, f)


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Prepare dataset for training.")
    parser.add_argument("--ds_src", type=str, required=True, help="Path to the raw dataset source directory.")
    parser.add_argument("--ds_dest", type=str, required=True,
                        help="Path to the processed dataset destination directory.")
    parser.add_argument("--ds_name", type=str, required=True, help="Name of the dataset.")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()
    main(args.ds_src, args.ds_dest, args.ds_name)
