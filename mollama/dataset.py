"""Module for processing ChEBI-20 data into molecule captioning/generation datasets.

This module provides functionalities to convert ChEBI-20 data into datasets suitable for
molecule captioning and generation tasks. It contains functions to format data rows into
molecule captioning and generation instructions, create datasets, and convert ChEBI-20 data
into specific dataset formats.

Functions:
- _format_to_molecule_captioning_instruction(row: Dict[str, str]) -> str: Formats a data row into
    a molecule captioning instruction.
- _format_to_molecule_generation_instruction(row: Dict[str, str]) -> str: Formats a data row into
    a molecule generation instruction.
- _create_molecule_captioning_dataset(file_path: str) -> Dataset: Creates a molecule captioning dataset
    from a provided file path.
- _create_molecule_generation_dataset(file_path: str) -> Dataset: Creates a molecule generation dataset
    from a provided file path.
- convert_chebi_20_data_to_molecule_captioning_dataset(train_data_path: str, val_data_path: str,
    test_data_path: str) -> DatasetDict: Converts ChEBI-20 data to a molecule captioning dataset.
- convert_chebi_20_data_to_molecule_generation_dataset(train_data_path: str, val_data_path: str,
    test_data_path: str) -> DatasetDict: Converts ChEBI-20 data to a molecule generation dataset.

Example Usage:
    python dataset.py --train_data_path path/to/train_data.txt --val_data_path path/to/val_data.txt
                      --test_data_path path/to/test_data.txt --task_name smile2caption
                      --save_path path/to/save_directory
"""
import csv
import argparse
from typing import Dict
from datasets import Dataset, DatasetDict


def _format_to_molecule_captioning_instruction(row: Dict[str, str]) -> str:
    """Formats a provided data row to molecule captioning instruction.

    Args:
        row (Dict[str, str]): A dictionary containing keys "SMILES" and "description"
            representing the molecule's SMILES string and its description, respectively.

    Returns:
        str: A formatted string containing the molecule captioning instruction.
            The instruction includes the SMILES information and a request for a detailed description
            of the molecule.

    Example:
        >>> row_data = {"cid": "1234", "SMILES": "C1CCCCC1", "description": "This is a cyclic molecule."}
        >>> _format_to_molecule_captioning_instruction(row_data)
        '<s>[INST] Molecule SMILES: C1CCCCC1. Given the molecule SMILES your task is to provide a detailed description of the molecule. [/INST] This is a cyclic molecule.'
    """
    return f'<s>[INST] Molecule SMILES: {row["SMILES"]}. Given the molecule SMILES your task is to provide a detailed description of the molecule. [/INST] {row["description"]}'


def _format_to_molecule_generation_instruction(row: Dict[str, str]) -> str:
    """Formats a provided data row to molecule generation instruction from SMILES string.

    Args:
        row (Dict[str, str]): A dictionary containing keys "SMILES" and "description"
            representing the molecule's SMILES string and its description, respectively.

    Returns:
        str: A formatted string containing the molecule generation instruction.
            The instruction includes the molecule description and a request for generating
            an accurate SMILES string for the molecule.

    Example:
        >>> row_data = {"cid": "1234", "SMILES": "C1CCCCC1", "description": "This is a cyclic molecule."}
        >>> _format_to_molecule_generation_instruction(row_data)
        '<s>[INST] Molecule description: This is a cyclic molecule. Given the molecule description your task is to generate an accurate SMILES string of the molecule. [/INST] C1CCCCC1'
    """
    return f'<s>[INST] Molecule description: {row["description"]}. Given the molecule description your task is to generate an accurate SMILES string of the molecule. [/INST] {row["SMILES"]}'


def _create_molecule_captioning_dataset(file_path: str) -> Dataset:
    """Creates a molecule captioning dataset from the provided file path.

    Args:
        file_path (str): Path to the dataset file containing molecule information
            with columns delimited by tabs. Expected columns are "SMILES" and "description".

    Returns:
        Dataset: A dataset created with instructions formatted for molecule captioning.

    Example:
        >>> dataset_path = "path/to/molecule_captioning_data.txt"
        >>> captioning_dataset = _create_molecule_captioning_dataset(dataset_path)
    """
    data = []
    with open(file_path) as fp:
        csv_reader = csv.DictReader(fp, delimiter="\t")
        for row in csv_reader:
            row["instruction"] = _format_to_molecule_captioning_instruction(row)
            data.append(row)
    return Dataset.from_list(data)


def _create_molecule_generation_dataset(file_path: str) -> Dataset:
    """Creates a molecule generation dataset from the provided file path.

    Args:
        file_path (str): Path to the dataset file containing molecule information
            with columns delimited by tabs. Expected columns are "SMILES" and "description".

    Returns:
        Dataset: A dataset created with instructions formatted for molecule generation.

    Example:
        >>> dataset_path = "path/to/molecule_generation_data.txt"
        >>> generation_dataset = _create_molecule_generation_dataset(dataset_path)
    """
    data = []
    with open(file_path) as fp:
        csv_reader = csv.DictReader(fp, delimiter="\t")
        for row in csv_reader:
            row["instruction"] = _format_to_molecule_generation_instruction(row)
            data.append(row)
    return Dataset.from_list(data)


def convert_chebi_20_data_to_molecule_captioning_dataset(
        train_data_path: str,
        val_data_path: str,
        test_data_path: str,
) -> DatasetDict:
    """Converts ChEBI-20 data to a molecule captioning dataset.

    Args:
        train_data_path (str): Path to the training data file.
        val_data_path (str): Path to the validation data file.
        test_data_path (str): Path to the test data file.

    Returns:
        DatasetDict: A dictionary containing train, validation, and test datasets
        formatted for molecule captioning.

    Example:
        >>> train_path = "path/to/train_data.txt"
        >>> val_path = "path/to/validation_data.txt"
        >>> test_path = "path/to/test_data.txt"
        >>> captioning_dataset = convert_chebi_20_data_to_molecule_captioning_dataset(train_path, val_path, test_path)
    """
    train_data = _create_molecule_captioning_dataset(train_data_path)
    val_data = _create_molecule_captioning_dataset(val_data_path)
    test_data = _create_molecule_captioning_dataset(test_data_path)

    return DatasetDict(
        train=train_data,
        validation=val_data,
        test=test_data,
    )


def convert_chebi_20_data_to_molecule_generation_dataset(
        train_data_path: str,
        val_data_path: str,
        test_data_path: str,
) -> DatasetDict:
    """Converts ChEBI-20 data to a molecule generation dataset.

    Args:
        train_data_path (str): Path to the training data file.
        val_data_path (str): Path to the validation data file.
        test_data_path (str): Path to the test data file.

    Returns:
        DatasetDict: A dictionary containing train, validation, and test datasets
        formatted for molecule generation.

    Example:
        >>> train_path = "path/to/train_data.txt"
        >>> val_path = "path/to/validation_data.txt"
        >>> test_path = "path/to/test_data.txt"
        >>> generation_dataset = convert_chebi_20_data_to_molecule_generation_dataset(train_path, val_path, test_path)
    """
    train_data = _create_molecule_generation_dataset(train_data_path)
    val_data = _create_molecule_generation_dataset(val_data_path)
    test_data = _create_molecule_generation_dataset(test_data_path)

    return DatasetDict(
        train=train_data,
        validation=val_data,
        test=test_data,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to convert ChEBI-20 data to molecule captioning/generation datasets"
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="Path to train data"
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        help="Path to validation data"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        help="Path to test data"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="smile2caption",
        choices=["smile2caption", "caption2smile"],
        help="Task name: smile2caption or caption2smile"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save the instruction dataset"
    )

    args = parser.parse_args()

    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    test_data_path = args.test_data_path
    task_name = args.task_name
    save_path = args.save_path

    if task_name == "smile2caption":
        chebi_20_data = convert_chebi_20_data_to_molecule_captioning_dataset(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            test_data_path=test_data_path,
        )
    elif task_name == "caption2smile":
        chebi_20_data = convert_chebi_20_data_to_molecule_generation_dataset(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            test_data_path=test_data_path,
        )
    else:
        raise ValueError("Invalid task name. Use 'smile2caption' or 'caption2smile'.")

    chebi_20_data.save_to_disk(f"{save_path}/ChEBI-20_{task_name}_data")
