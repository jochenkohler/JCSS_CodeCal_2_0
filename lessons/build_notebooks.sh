#!/usr/bin/env bash
# Convert the lessons to executed Jupyter notebooks (outputs and figures
# included), so they can be read without running anything.
#
#   cd lessons && bash build_notebooks.sh          # all lessons, ~8 minutes
#   cd lessons && bash build_notebooks.sh 01 02    # selected ones
#
# Needs: pip install jupytext nbconvert ipykernel
set -e
cd "$(dirname "$0")"
mkdir -p notebooks
if [ $# -eq 0 ]; then set -- 01 02 03 04 05 06; fi
for n in "$@"; do
  src=$(ls "${n}"_*.py)
  echo "executing $src ..."
  jupytext --to ipynb --execute --run-path . "$src" -o "notebooks/${src%.py}.ipynb"
done
echo "notebooks written to lessons/notebooks/"
