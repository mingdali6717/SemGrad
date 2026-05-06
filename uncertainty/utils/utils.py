import pandas as pd
import os
import json
import functools
from datasets import load_dataset
from loguru import logger

def load_data(path):
    extension = path.split(".")[-1]
    if extension == "txt":
        extension = "text"
    elif extension == "jsonl":
        extension = "json"

    return load_dataset(extension, data_files=path)["train"]



def get_gpu_memory():
    # Runs the nvidia-smi command and retrieves GPU memory usage
    result = os.popen("nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader").readline().strip()
    return float(result)


def reshape_sequences(sequences, n):
    """
    reshape input sequences List[] with len m*n to List[List[]],each sublist is with len n
    """
    assert n == int(n), f"n shoud be an interger, but {n:.2f} is given"
    prompts = []
    n = int(n)
    assert len(
        sequences) % n == 0, f"length of sequences should be a multiple of {n}, but the length of given sequences is {len(sequences)}"
    m = int(len(sequences) / n)
    start_id = 0
    for _ in range(m):
        end_id = start_id + n
        prompts.append(sequences[start_id: end_id])
        start_id = end_id
    return prompts

def normalize_quotes(s: str) -> str:
    # maps U+2018, U+2019, U+2032  → ASCII apostrophe '
    return s.translate(str.maketrans({
        "\u2018": "'",   # ‘
        "\u2019": "'",   # ’
        "\u2032": "'",   # ′
        "`":  "'",
        "“": "\"",
        "”": "\""
    }))

def normalize_cache_path(cache_path, save_path, method_name, file_name):
    if cache_path is not None:
        if os.path.isdir(cache_path):
            
            normalize_cache_path = os.path.join(cache_path, file_name)
        else:
            assert cache_path.endswith(".json"), f"cached file should be end with '.json', but '{cache_path}' is given"
            normalize_cache_path = cache_path
    else:
        normalize_cache_path = None

    if save_path is not None:

        if os.path.isdir(save_path):
            normalize_save_path = os.path.join(save_path, file_name)
        else:
            assert save_path.endswith(".json"), "save file should be end with '.json'"
            normalize_save_path = save_path  
    elif normalize_cache_path is not None:
        normalize_save_path = normalize_cache_path
    else:
        normalize_save_path = None

    if normalize_cache_path is not None and (not os.path.exists(normalize_cache_path)):
        logger.info(f"cached path {normalize_cache_path} does not exist, evaluate {method_name} from scratch.")
        normalize_cache_path = None
    
    return normalize_cache_path, normalize_save_path



