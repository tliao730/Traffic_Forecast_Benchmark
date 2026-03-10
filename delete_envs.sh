#!/bin/bash

# This script deletes all .venv directories in the current directory and its subdirectories.
find . -type d -name ".venv" -exec rm -rf {} +
echo "All .venv directories have been deleted."