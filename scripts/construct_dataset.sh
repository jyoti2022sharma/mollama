#!/bin/bash

# Define paths to dataset.py (should be found at mollama/dataset.py) and Python executable
PYTHON_EXECUTABLE="python"  # Replace with your Python executable if needed
SCRIPT_PATH="path/to/dataset.py"

# Define dataset paths
TRAIN_DATA_PATH="path/to/train_data.txt"
VAL_DATA_PATH="path/to/val_data.txt"
TEST_DATA_PATH="path/to/test_data.txt"

# Define task name and save directory path
TASK_NAME="smile2caption" # the other option is caption2smile
SAVE_PATH="path/to/save_directory"

# Execute the Python script with specified arguments
$PYTHON_EXECUTABLE $SCRIPT_PATH \
    --train_data_path $TRAIN_DATA_PATH \
    --val_data_path $VAL_DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --task_name $TASK_NAME \
    --save_path $SAVE_PATH
