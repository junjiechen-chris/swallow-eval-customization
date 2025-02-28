#!/usr/bin/env python
# -*- coding:utf-8 -*-

import re
from typing import Dict, Any, List
import json
import argparse
import os
import pandas as pd
from collections import defaultdict
from functools import reduce
from operator import getitem


# TODO: Hardcorded subtasks columns map for averaging the scores
SUBTASKS_COLUMNS_MAP = {
    "MC": ["jcommonsenseqa"],
    "NLI": ["jamp (NLI)", "janli (NLI)", "jnli", "jsem", "jsick (NLI)"],
    "QA": ["jemhopqa", "niilc"],
    "RC": ["jsquad"],
}


def load_json(input_path: str) -> Dict[str, Any]:
    with open(input_path, "r") as file:
        data = json.load(file)
    return data


def get_nested_dict_value(input_path: str, keys: List[str]) -> float:
    # get value from nested dictionary by key string
    # e.g. get_dict_value(data, [key1, key2, key3, ...])
    d = load_json(input_path)
    try:
        metric = float(reduce(getitem, keys, d))
    except KeyError:
        print(f"Key not found: {keys}")
        return -1.0

    return metric


def get_average_score(input_path: str, keys: List[str]) -> float:
    """get average score from multiple keys"""
    scores = [get_nested_dict_value(input_path, key) for key in keys]

    # if scores has -1.0, return -1.0
    # because some of the subtasks are not evaluated
    if -1.0 in scores:
        return -1.0

    return sum(scores) / len(scores)


def aggregate_results(
    model: str,
    result_dir: str,
    column_path_key_path: str,
) -> Dict[str, float]:
    """load all results of the model and aggregate the scores into a single dictionary"""
    column_path_key_csv = pd.read_csv(column_path_key_path)

    task_keys_map = {
        k: [key.replace("MODEL_NAME", model.replace("/", "_")) for key in v.split(".")]
        for k, v in column_path_key_csv[["column", "key"]].values
    }

    results = {}
    overall = []

    for _, row in column_path_key_csv.iterrows():
        column, path, _, max_score = row
        keys = task_keys_map[column]
        input_path = os.path.join(result_dir, path)

        # defalut value if the column is empty
        metric = -1.0
        # error handling if the file is not found
        if not os.path.exists(input_path):
            print(f"Column: {column} is empty")
            print(f"File not found: {input_path}")
            results[column] = metric
            overall.append(metric)
            continue

        if column in SUBTASKS_COLUMNS_MAP.keys():
            subtasks = SUBTASKS_COLUMNS_MAP[column]
            keys_subtasks = [task_keys_map[subtask] for subtask in subtasks]
            metric = get_average_score(input_path, keys_subtasks)
        else:
            try:
                metric = get_nested_dict_value(input_path, keys)
            except:
                print("Unfound path:", input_path)
                print("Unfound keys:", keys)

        # Normalize the score using predefined max_score
        metric = metric / max_score

        results[column] = metric
        overall.append(metric)

    json_result = {
        "model": model,
        "scores": results,
        "overall": ",".join(map(str, overall)),
        "tasks": list(results.keys()),
    }

    json.dump(
        json_result,
        open(f"{result_dir}/result.json", "w"),
        indent=2,
        ensure_ascii=False,
    )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, type=str, help="Model name to aggregate"
    )
    tmp_args, _ = parser.parse_known_args()
    parser.add_argument(
        "--result-dir",
        type=str,
        default=os.path.join("results", tmp_args.model),
        help="Result directory",
    )
    parser.add_argument(
        "--column-path-key-path",
        type=str,
        default="scripts/column-path-key.csv",
        help="Path to column-path-key.csv",
    )
    args = parser.parse_args()
    aggregate_results(
        args.model,
        result_dir=args.result_dir,
        column_path_key_path=args.column_path_key_path,
    )


if __name__ == "__main__":
    main()
