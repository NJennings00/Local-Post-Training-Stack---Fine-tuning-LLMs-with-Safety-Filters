# tests/conftest.py

import sys
from os.path import abspath, dirname, join

# Get the path to the root of the project
project_root = abspath(join(dirname(__file__), '..'))

# Insert the project root into the system path
# This allows 'data' and other top-level directories to be imported by tests
sys.path.insert(0, project_root)