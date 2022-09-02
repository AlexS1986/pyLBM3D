#!/usr/bin/env bash

echo "Running pre-commit hook"
#./scripts/run-tests.bash
python ./test/TestCore.py

# $? stores exit value of the last command
if [ $? -ne 0 ]; then
 echo "Tests must pass before commit!"
 exit 1
fi
