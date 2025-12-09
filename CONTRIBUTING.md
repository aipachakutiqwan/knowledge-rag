# CONTRIBUTING


Welcome to this repository, and thank you for your interest in contributing!

This guide is chiefly for users wishing to contribute to the project.


## ⚡ Development

Below are the tools used during development. Please ensure that you always **start your contributions by creating a feature branch from the "main" branch.**

:exclamation: Note: Ensure that **Python version 3.12.x** is installed on your local machine, as it is the version compatible with this documentation.

#### 🌱 Managing dependencies

Based on [UV](https://docs.astral.sh/uv/getting-started/installation/)

- Install UV Package manager
    ```bash
    pip install uv
    ```
- Create Python environment
    ```bash
    uv venv --python 3.12
    ```
- Activate environment
    ```bash
    source .venv/bin/activate
    ```
- Install Pytorch and Pytorch Geometric libraries for GPU (CUDA 12.4)
    ```bash
    uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    uv pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.6.0+cu124.html
    uv pip install torch_sparse -f https://pytorch-geometric.com/whl/torch-2.6.0+cu124.html
    uv pip install pyg_lib torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
    ```
- If you intend to run using CPU, install Pytorch and Pytorch Geometric libraries for CPU. Running on a CPU is only recommended for use with small or sample datasets.
    ```bash
    uv pip install torch torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
    uv pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.6.0+cpu.html
    uv pip install torch_sparse -f https://pytorch-geometric.com/whl/torch-2.6.0+cpu.html
    uv pip install pyg_lib  torch_cluster torch_spline_conv -f https://pytorch-geometric.com/whl/torch-2.6.0+cpu.html
    ```
- Install Python libraries
    ```bash
    uv pip install -r pyproject.toml --group dev
    ```

#### 🌱 Initializing pre-commit

Pre-commit settings are available in `.pre-commit-config.yaml`

- [Install pre-commit](https://pre-commit.com/#install)
    ```bash
    pip install pre-commit
    ```
- Verify pre-commit version: currently using  4.0.1
    ```bash
    pre-commit --version
    ```
- Run pre-commit.
    ```bash
    pre-commit run
    ```

#### 🌱 Versioning with Commitizen


Based on [Commitizen](https://commitizen-tools.github.io/commitizen/)

Settings are available in `pyproject.toml`

- Write clear and meaningful commit messages following the standard [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). Each commit should include the following structural elements to clearly communicate the intent to other developers and consumers of the code.

- **Always commit** using the commitizen helper.
    ```bash
    cz commit
    ```
- **Always bump version** updating changelog file.
    ```bash
     cz bump --changelog
    ```

#### 🌱 Run Poe tasks

Settings are available in `poe.toml`

Based on [Poe The Poet](https://poethepoet.natn.io/index.html)

- Check what tasks are available.
    ```bash
    poe
    ```
- Run an specific task (e.g. linting code).
    ```bash
    poe lint
    ```

## ⚡ How create pull requests

Here are the steps to create a Pull Request.

- Fork the repository and make your code contributions in a dedicated feature branch (**created from "main" branch**).
- Run the linter to check the code content.
    ```bash
    poe lint
    ```
- Run unit tests and ensure there is no reduction in test coverage by checking htmlcov/index.html
    ```bash
    poe test
    ```
- Commit your changes using Commitizen for standardized commit messages.
    ```bash
    cz commit
    ```
- If necessary, create a new release version and bump the version number.
    ```bash
    cz bump --changelog
    ```
- If the previous step was successful, push your code and create a pull request to the "main" branch for code review.


