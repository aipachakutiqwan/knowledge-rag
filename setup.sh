#!/usr/bin/bash

pip install uv

uv venv --python 3.12
source .venv/bin/activate

uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.6.0+cu124.html
uv pip install torch_sparse -f https://pytorch-geometric.com/whl/torch-2.6.0+cu124.html
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
uv pip install -r pyproject.toml --group dev

