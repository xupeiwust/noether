<div align="center">

![noether-framework-logo](./docs/source/_static/noether-framework-logo.svg)
# Noether Framework

[![Docs - noether-docs.emmi.ai](https://img.shields.io/static/v1?label=Docs&message=noether-docs.emmi.ai&color=2ea44f&logo=gitbook)](https://noether-docs.emmi.ai)
[![License: Custom](https://img.shields.io/badge/License-Custom-orange.svg)](./LICENSE.txt)
[![Static Badge](https://img.shields.io/badge/Walkthrough-Tutorial-DD537C)](./tutorial/README.MD)

[![Tests](https://github.com/Emmi-AI/noether/actions/workflows/run-tests.yml/badge.svg)](https://github.com/Emmi-AI/noether/actions/workflows/run-tests.yml)

</div>

<div>
<strong>Noether</strong> is Emmi AI’s <strong>open software framework for Engineering AI</strong>. Built on 
<strong>modern transformer building blocks</strong>, it delivers the full engineering stack, allowing teams to build, 
train, and operate industrial simulation models across engineering verticals, eliminating the need for component 
re-engineering or an in-house deep learning team.
</div>

## Key Features

- **Modular Transformer Architecture:** Built on modern building blocks optimized for physical systems.
- **Hardware Agnostic:** Seamless execution across CPU, MPS (Apple Silicon), and NVIDIA GPUs.
- **Industrial Grade:** Designed for high-fidelity industrial simulations and engineering verticals.
- **Ready for Scale:** Built-in support for Multi-GPU and SLURM cluster environments.

---

# Table of contents

- [Installation](#installation)
  - [Pre-requisites](#pre-requisites)
  - [Working with the source code](#working-with-the-source-code)
    - [How to clean up and do a fresh installation](#how-to-cleanup-and-do-a-fresh-installation)
  - [Working with pre-built packages](#working-with-pre-built-packages)
- [NVIDIA GPUs and Linux setup](#newest-nvidia-gpus-and-linux-setup)
- [Quickstart and hardware allocation](#quickstart-and-hardware-allocation)
  - [Training via SLURM environment](#training-via-slurm-environment)
  - [Training in a non-managed environment](#training-in-a-non-managed-environment)
- [Performance Benchmarks](#performance-benchmarks)
  - [Observations](#observations)
- [Contributing](#contributing)
  - [Guidelines](#guidelines)
  - [Third-party contributors](#third-party-contributors)
  - [Configuring IDEs](#configuring-ides)
- [Supported systems](#supported-systems)
- [Licensing](#licensing)
- [Citing](#citing)

---
# Installation

It is possible to use the framework either from source or from the pre-built packages.

## Pre-requisites

- install [uv](https://docs.astral.sh/uv/getting-started/installation/) as the package manager on your system
- clone the repo into your desired folder: `git clone git@github.com:Emmi-AI/noether.git`
- follow the next steps 🚀

## Working with the source code

1. Create a fresh virtual environment and synchronize the core dependencies:

```console
uv venv && source .venv/bin/activate
uv sync
```

**Note:** Initial installation may take several minutes as third-party dependencies are compiled. Duration depends on 
your hardware and network speed.

2. Build hardware-specific dependencies
  Because `torch-scatter` and `torch-cluster` must link directly to your local hardware drivers (CUDA on Linux or MPS 
  on Mac), you must force a local build:

```console
uv pip install --no-binary torch-cluster,torch-scatter torch-cluster torch-scatter --force-reinstall
```

> [!IMPORTANT]
> This step is optional as in the clean installation `uv sync` should build the packages for you. 
> 
> It is **important** to make sure that your environment has necessary tools 
> ([see below](#newest-nvidia-gpus-and-linux-setup)).

Validate your installation by simply running the tests (if something fails with module import errors it means that the 
installation was incomplete):
```console
pytest -q tests/
```
if the tests are passed (warnings are okay to be logged) then you're all set and ready to go!

### How to clean up and do a fresh installation

You might be in a situation when your venv won't be configured as intended anymore, to fix this:

- Deactivate existing environment in your terminal by running: `deactivate`
- Remove existing `.venv` (optionally add `uv.lock`): `rm -rf .venv uv.lock`
- [Optional] Clean uv cache: `uv cache clean`
- Create a new venv and activate it: `uv venv && source .venv/bin/activate`
- [Optional] If deleted, generate a new `uv.lock` file: `uv lock`
- [Optional] If contributor: `pre-commit install`

## Working with pre-built packages
> [!NOTE]
> To be added later.

---
# Newest NVIDIA GPUs and Linux setup

Due to third-party dependencies, like `torch_geometric`, `torch_scatter`, etc., there is a need for a few extra steps
to make it working on Linux. 

1. Make sure that you have installed relevant CUDA toolkit (here we use 12.9 to make sure that NVIDIA Blackwell 
architecture is supported):

```console
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-9
```

2. Create a symlink for your fresh CUDA toolkit installation to a generic path (make sure to link your target version!):

```console
sudo ln -s /usr/local/cuda-12.9 /usr/local/cuda
```

3. Set environment variables to help `uv` to build third-party packages with the newest CUDA:

```console
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
```

Now you should be able to test `nvcc`:

```console
which nvcc

# outputs something like:
# /usr/local/cuda/bin/nvcc
```

```console
nvcc --version

# will log something like this:
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_May_27_02:21:03_PDT_2025
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0
```

```console
uv pip install --no-binary torch-cluster,torch-scatter torch-cluster torch-scatter --force-reinstall
```

---
# Quickstart and hardware allocation

There are many possible hardware configurations that one can have:

- CPU
- MPS (Apple silicon)
- x1 GPU
- multiple GPUs
- multiple nodes

Configuring these can be tedious, and to make sure that you're up-to-speed in no time let's dive into the CLI examples.

## Training on CPU and/or MPS (Apple silicon)

We use Macbooks for development, and sometimes it's simply convenient to run code locally to validate its correctness.
We will use the `tutorial/` with `Shapenet-Car` dataset as an example of how to configure neatly. For more explanation 
about the dataset and respective commands - check the `tutorial/README.md`.

This approach should be valid for other systems too (CPU), but it will lead to long execution times.

```console
uv run noether-train --hp tutorial/configs/train_shapenet.yaml \
    +experiment/shapenet=upt \
    dataset_root=/Users/user/data/shapenet_car \
    trainer.precision=fp32 \
    +accelerator=mps \  # << here simply use "mps" or "cpu" without the quotes
    tracker=disabled \
    +seed=1
```

## Training via SLURM environment

SLURM helps with hardware allocation in constrained environments, like shared clusters. We assume that you are familiar
with how to allocate your resources and will show what to do after you have access to the hardware.

```console
uv run noether-train --hp tutorial/configs/train_shapenet.yaml \
    +experiment/shapenet=upt \
    dataset_root=/Users/user/data/shapenet_car \
    trainer.precision=fp32 \  # << on NVIDIA GPUs this can be set to various options: fp32, fp16, bf16, etc.
    +accelerator=gpu \  # << note that we switched this to "gpu" (no quotes)
    tracker=disabled \
    +seed=1
```

This command, by default, will utilize ALL available GPUs. This comes in handy when SLURM is used as your main 
orchestrator.

## Training in a non-managed environment
Non-managed environment is a simple definition of a system where you have access to all of your hardware, like 
a personal workstation. You can just run the same command above - this will use **all of your GPUs**. For more 
granularity in your setting you can define `devices` argument:

```console
uv run noether-train --hp tutorial/configs/train_shapenet.yaml \
    +experiment/shapenet=upt \
    dataset_root=/Users/user/data/shapenet_car \
    trainer.precision=fp32 \
    +accelerator=gpu \
    +devices=\"0\"  \  # << note that we explicitely cast value to a string as Hydra tends to convert values to integers
    tracker=disabled \
    +seed=1
```

Here we will run training on a single GPU that has an ID equal to 0. If your workstation has more GPUs you can declare
`devices` string as: `\"0,1,2,3\"`, `\"0,2\"`, `\"1,3\"`, etc. Just make sure that you have `\` in front of each `"`.

---
# Performance Benchmarks

The following benchmarks demonstrate **Noether**'s scaling capabilities using the `Shapenet-Car` dataset 
and the `AB-UPT` model. 

> [!NOTE]
> All benchmarks were conducted using **FP32 precision** to establish a baseline for raw computational performance.

| Hardware | Config | Precision | Time | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **MacBook Pro M3 Max** | 1x MPS | FP32 | 135m | 1.0x |
| **RTX Pro 4500 (Blackwell)** | 1x GPU | FP32 | 26m | 5.2x |
| **RTX Pro 4500 (Blackwell)** | 2x GPU | FP32 | 8m | 16.8x |
| **NVIDIA H100** | 1x GPU | FP32 | 5.7m | 23.6x |


## Observations

- **Linear+ Scaling:** The transition from 1x to 2x Blackwell GPUs shows super-linear scaling, highlighting efficient 
memory management and low DDP overhead.
- **Local to Cluster:** Engineers can prototype locally on Apple Silicon and transition to H100 clusters with zero code 
changes, benefiting from a ~24x performance increase.

---
# Contributing

## Guidelines

We follow these standards:

- Use typed coding in Python.
- Write documentation to new features and modules:
  - In case of larger modules make sure to update the documentation that is not autogenerated under the `docs/`.
  - For smaller features writing a clear API documentation is enough and required.
- Before committing your changes:
  - Run tests via `pytest -q tests/`.
  - Ensure that [pre-commit hooks](#pre-commit-hooks) are not disabled and are runnable at every commit.
    We are using `ruff` as a linter and formatter as well as `mypy` for type checking.
    Their configuration is defined in the project's root [pyproject.toml](pyproject.toml).
- Creating pull requests (PRs) is a mandatory step for any incoming changes that will end up on the `main` branch.
  - For a PR to be merged at least one core maintainer must give their approval.
  - All test must be green

## Pre-commit Hooks

To install pre-commit execute:
```console
pre-commit install
```
To run the pre-commit configuration on all files, you can use:
```console
pre-commit run --all-files
```
To run the pre-commit configuration on specific files use:
```console
pre-commit run --files /your/file/path1.py /your/file/path2.py
```

## Third-party contributors

In case of bugs use a corresponding template to create an issue.

In case of feature requests you can submit a PR with clear description of the proposed feature. In that case it must 
follow the [guidelines](#guidelines), or file a feature request as an issue. In that case, we will consider adding it to our 
backlog.

## Configuring IDEs

Emmi AI developers have to consult the internal documentation for more granular setup.

### Pycharm

- Mark `src/` directory as `Sources Root` (right mouse button click on the folder -> `Mark Directory as`)
- Settings -> Editor -> Code Style -> Python -> Tabs and Indents -> change `Continuation indent` from 8 to 4.
- Settings -> Editor -> Code Style -> Python -> Spaces -> Around Operators -> `Power operator (**)`

---
# Working with GitHub

With available GitHub Actions we automate several workflows relevant to our development ranging from buildings the docs
to building our modules as wheel files.

To test the desired workflow locally it is recommended to use [act](https://github.com/nektos/act).

> [!NOTE]
> Make sure to install Docker Desktop as requested by the official documentation.

Install it on a Mac with: `brew install act`

For example, to check the package release pipeline:
```console
act workflow_dispatch --input version_type=patch -W .github/workflows/release.yml
```
or to see if tests are runnable:
```console
act pull_request -W .github/workflows/run-tests.yml
```

---
# Supported systems
Worth noting that we work with macOS and Linux environments thus in case of any issues on Windows, at this time, you 
have to find workarounds yourself.

---
# Licensing

**Noether** is released under a custom license to balance open research with sustainable development. We encourage 
free use for research and academic exploration. However, a commercial license or explicit permission is required for 
distribution, integration into commercial products, or using pre-trained models for profit. 

Please see [LICENSE.txt](./LICENSE.txt) for the full legal terms.

---
# Citing

If you use **Noether** in your research or industrial applications, please cite this repository. 
A formal BibTeX entry for our forthcoming ArXiv publication will be provided here shortly.

```bibtex
@misc{noether2026,
  author = { Bleeker, Maurits AND Hennerbichler, Markus AND Kuksa, Pavel },
  title = {Noether: A PyTorch-based Framework for Engineering AI},
  year = {2026},
  publisher = {GitHub},
  note = {Equal contribution},
  url = {https://github.com/Emmi-AI/noether}
}
```
