#!/bin/bash
# perform test
SCRIPT_DIR=$( dirname "$0" )

pytest -sv $SCRIPT_DIR/../../tests/