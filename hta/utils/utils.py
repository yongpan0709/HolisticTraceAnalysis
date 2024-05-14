# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
from enum import Enum
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import psutil
import os
import sys
import shutil
from datetime import datetime


class KernelType(Enum):
    COMMUNICATION = 0
    MEMORY = 1
    COMPUTATION = 2


class IdleTimeType(Enum):
    HOST_WAIT = 0
    KERNEL_WAIT = 1
    OTHER = 2


def normalize_path(path: str) -> str:
    """
    Convert a Linux path to Python path.

    Args:
        path (str) : a path acceptable by the OS.

    Returns:
        A path supported by Python.
    """
    if path.startswith("./"):
        path2 = path[2:]
        if len(path2) > 0:
            normalized_path = str(Path.cwd().joinpath(path2))
        else:
            normalized_path = str(Path.cwd())
    elif path.startswith("~/"):
        path2 = path[2:]
        if len(path2) > 0:
            normalized_path = str(Path.home().joinpath(path2))
        else:
            normalized_path = str(Path.home())
    else:
        normalized_path = path
    return normalized_path


def is_comm_kernel(name: str) -> bool:
    """
    Check if a given GPU kernel is a communication kernel.

    Args:
        name (str): name of the GPU kernel.

    Returns:
        A boolean indicating if the kernel is a communication kernel.
    """
    return "ncclKernel" in name


def is_memory_kernel(name: str) -> bool:
    """
    Check if a given GPU kernel is a memory kernel.

    Args:
        name (str): name of the GPU kernel.

    Returns:
        A boolean indicating if the kernel is an IO kernel.
    """
    return "Memcpy" in name or "Memset" in name


def get_kernel_type(name: str) -> str:
    if is_comm_kernel(name):
        return KernelType.COMMUNICATION.name
    elif is_memory_kernel(name):
        return KernelType.MEMORY.name
    else:
        return KernelType.COMPUTATION.name


def get_memory_kernel_type(name: str) -> str:
    """Memcpy Type is basically a prefix of the kernel name ~ Memcpy DtoH"""
    if name[:6] == "Memset":
        return "Memset"
    if name[:6] != "Memcpy":
        return "Memcpy Unknown"
    prefix_size = 11  # len("Memcpy DtoH")
    return name[:prefix_size]


def merge_kernel_intervals(kernel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all kernel intervals in the given dataframe such that there are no overlapping.
    """
    kernel_df.sort_values(by="ts", inplace=True)
    kernel_df["end"] = kernel_df["ts"] + kernel_df["dur"]
    # Operators within the same group need to be merged together to form a larger interval.
    kernel_df["group"] = (kernel_df["ts"] > kernel_df["end"].shift().cummax()).cumsum()
    kernel_df = (
        kernel_df.groupby("group", as_index=False)
        .agg({"ts": "min", "end": "max"})
        .drop(["group"], axis=1)
        .sort_values(by="ts")
    )
    return kernel_df


def shorten_name(name: str) -> str:
    """Shorten a long operator/kernel name.

    The CPU operator and CUDA kernel name in the trace can be too long to follow.
    This utility removes the functional arguments, template arguments, and return values
    to make the name easy to understand.
    """
    s: str = name.replace("->", "")
    stack: List[str] = []
    for c in s:
        if c == ">":  # match generic template arguments
            while len(stack) and stack[-1] != "<":
                stack.pop()

            if len(stack) > 0 and stack[-1] == "<":
                stack.pop()
        elif c == ")":  # match arguments or comments
            while len(stack) and stack[-1] != "(":
                stack.pop()
            if len(stack) > 0 and stack[-1] == "(":
                stack.pop()
        else:
            stack.append(c)
    return "".join(stack).split(" ")[-1]


def flatten_column_names(df: pd.DataFrame) -> None:
    """Flatten a DataFrame's a multiple index column names to a single string"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).rstrip("_") for col in df.columns]


def get_mp_pool_size(obj_size: int, num_objs: int) -> int:
    """
    Estimate the maximum pool size for multiprocessing

    Args:
        obj_size (int): the size of objects to be processed
        num_objs (int): the total number of objects to be processed

    Returns:
        int
            the recommend pool size
    """
    free_mem = psutil.virtual_memory().available
    # Leave 20% buffer for system and other processes
    max_np = int(0.8 * free_mem / obj_size)
    return min(max_np, num_objs, mp.cpu_count())


def get_symbol_column_names(df: pd.DataFrame) -> Tuple[str, str]:
    """Get the proper column names for the `name` and `cat` attributes of string type in the DataFrame.

    Due to the encoding/decoding operations, it is impossible for a generic HTA routine to known which columns
    in a trace DataFrame have the symbol values for the events' `name` and `cat` attributes.

    Args:
        df (pd.DataFrame): A trace DataFrame.

    Returns:
        (column_name_for_name, column_name_for_cat)
    """
    name_column, cat_column = "", ""
    for column_name in ["name", "s_name"]:
        if column_name in df.columns and df.dtypes[column_name] == "object":
            name_column = column_name
            break
    for column_name in ["cat", "s_cat"]:
        if column_name in df.columns and df.dtypes[column_name] == "object":
            cat_column = column_name
            break
    return name_column, cat_column

import os
import shutil

def prepare_directory(directory_path, force_clear=False):
    """
    Prepares a directory for storing files. If the directory exists, it will be emptied; if it does not exist, it will be created.
    
    Parameters:
    - directory_path: The path to the target directory.
    """
    if os.path.exists(directory_path):
        if force_clear:
            # Empty the target directory
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        # Create the target directory
        os.makedirs(directory_path)

def find_and_copy_files(source_directory, target_directory, suffix_list):
    """
    Finds files in the specified directory whose names end with numbers from the given list, and copies them to a new directory.
    """
    # Prepare the target directory
    prepare_directory(target_directory)

    # Iterate through the source directory and copy matching files
    for filename in os.listdir(source_directory):
        name, ext = os.path.splitext(filename)
        for suffix in suffix_list:
            if name.endswith(f'_rank{suffix}') or name.startswith(f'worker{suffix}.'):
                # Filename matches, copy the file
                source_path = os.path.join(source_directory, filename)
                target_path = os.path.join(target_directory, filename)
                shutil.copy2(source_path, target_path)

def find_and_create_symlinks(source_directory, target_directory, suffix_list):
    """
    Finds files in the specified directory whose names end with numbers from the given list, and creates symbolic links for them in a new directory.
    """
    # Prepare the target directory
    prepare_directory(target_directory, force_clear=True)

    # Iterate through the source directory and create symbolic links for matching files
    for filename in os.listdir(source_directory):
        name, ext = os.path.splitext(filename)
        for suffix in suffix_list:
            if name.endswith(f'_rank{suffix}') or name.startswith(f'worker{suffix}.'):
                source_path = os.path.join(source_directory, filename)
                target_path = os.path.join(target_directory, filename)
                os.symlink(source_path, target_path)

def partition_files_across_directories(source_directory, target_directory, groups_list, skip=False):
    """
    Partitions files across directories based on a list of groups, creating symbolic links for each group in separate subdirectories.
    """
    all_sub_dirs = []
    for i, group in enumerate(groups_list):
        sub_dir = target_directory + f'_{i}'
        all_sub_dirs.append(sub_dir)
        if not skip:
            find_and_create_symlinks(source_directory, sub_dir, group)
    return all_sub_dirs

class LogToFile:
    """
    A context manager for redirecting stdout to a file.

    This class provides a mechanism for capturing the standard output
    to a specified file. It's useful for logging purposes, where the output
    of a block of code needs to be saved. If the file has not been written to
    before, it opens in write mode and logs the current timestamp. Otherwise,
    it appends to the existing file.

    Attributes:
        _has_cleared_files (dict):
            Tracks whether files have been cleared to avoid duplicate headers.
        filepath (str or None): Path to the log file. If None, stdout is not redirected.
        original_stdout (io.TextIOWrapper): Reference to the original stdout.

    """
    _has_cleared_files = {}  # Used to track whether each file has been cleared

    def __init__(self, filepath=None):
        """
        Initializes the context manager with the path to the log file.

        Args:
            filepath (str, optional): The path to the file where stdout will be redirected.
        """
        self.filepath = filepath
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def __enter__(self):
        """
        Enters the runtime context related to this object.

        The stdout is redirected to the specified file. If the file has not been
        written to before, it is cleared and initialized with a timestamp.

        Returns:
            LogToFile: The runtime context object.
        """
        if self.filepath:
            # Check if this file has already been cleared
            if self.filepath not in self._has_cleared_files:
                # If not, open in "w" mode to clear it and mark as cleared
                self.file = open(self.filepath, "w", encoding='utf-8')
                self.file.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S") + "\n")
                self._has_cleared_files[self.filepath] = True
            else:
                # If already cleared, open in "a" mode to append content
                self.file = open(self.filepath, "a", encoding='utf-8')
            sys.stdout = self.file
            sys.stderr = sys.stdout 
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the runtime context and restores the original stdout.

        The log file is closed if it was opened, and stdout is restored
        to its original state.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Traceback object.
        """
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self.filepath:
            self.file.close()