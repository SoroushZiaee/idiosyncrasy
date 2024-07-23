"""
    General project utilities
"""

import os, glob
import numpy as np

# import pyrallis
from dataclasses import dataclass, field, asdict
from typing import List
from colorama import Fore, Style
from multiprocessing.dummy import Pool
from PIL import Image
from tqdm import tqdm


def cprint1(s, *args, **kwargs):
    print(f"{Fore.CYAN}{Style.BRIGHT}{s}{Style.RESET_ALL}", *args, **kwargs)


def cprintc(s, *args, **kwargs):
    print(f"{Fore.CYAN}{s}{Style.RESET_ALL}", *args, **kwargs)


def cprintm(s, *args, **kwargs):
    print(f"{Fore.MAGENTA}{s}{Style.RESET_ALL}", *args, **kwargs)


def listify(value):
    """Ensures that the value is a list. If it is not a list, it creates a new list with `value` as an item."""
    if not isinstance(value, list):
        value = [value]
    return value


def get_images(images_paths, n_threads=20):
    """Return the list of frames given by list of absolute paths."""
    reader_fn = lambda image_path: np.array(Image.open(image_path).convert("RGB"))
    with Pool(n_threads) as pool:
        res_list = pool.map(reader_fn, images_paths)
    return res_list


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


class PoolReported:
    def __init__(self, n_threads):
        from multiprocessing.dummy import Pool

        self.pool = Pool(n_threads)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # Exception handling here
        self.pool.close()

    def map(self, f1, arg_list):
        with tqdm(total=len(arg_list)) as pbar:

            def f2(x):
                y = f1(x)
                pbar.update(1)
                return y

            return self.pool.map(f2, arg_list)
