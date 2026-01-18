# The Emmi-AI/core repository tutorial for the `ksuit` and `emmi` package

**Author**: Maurits Bleeker.

For questions, please reach out via slack in #team-research-code! 

## Introduction

Welcome to the `emmi.ai/core` repository tutorial! 

This tutorial will walk you through a (somewhat toy) project that is designed to teach you how to use the `ksuit` and `emmi` packages for building and training neural (CFD) surrogate models.
The `ksuit` package is our internal equivalent of PyTorch Lightning.
The `emmi` package contains modular (stable) components related to building and training neural surrogate models and data processing and collating that can be reused over multiple projects within the company.
git
Keep in mind that this tutorial is written *after* the `AB-UPT` project was finished, and that the code has been refactored a lot for this tutorial.
During research, project code is, of course, much more experimental, less maintained, and less structured. 
However, we would encourage you to keep the code as clean as possible during development.

This tutorial will be loosely based on the experiments run for section 4.4 in the [AB-UPT](https://arxiv.org/pdf/2502.09692) paper.
After the project was finished, I tried to unify all models we ran as much as possible.
That means, we use the same collator, trainer, and callback(s) for all models, and there is a **base model** class that all models inherit, which handles the input and out processing for all models, and has some other shared utilities.
That also implies that most models we run in the tutorial are not _exactly_ how they are either proposed in the original work or how we run them in the paper.

> [!IMPORTANT]  
> It is important to note that this tutorial does not result in the same numbers as reported in the AB-UPT paper. 
In the process of unifying all models, some models started to perform better or worse (especially due to the surface/volume conditioning and the change of input coordinate representation). 
Moreover, the results in the paper are the median over 5 runs with different seeds. 

> [!IMPORTANT]  
> The tutorial is written given the **current** version of ```emmi``` and ```ksuit ```.
In the near future, we will work on both packages a lot, and therefore, the tutorial will also change over time.

### File structure of the project

Our project uses the following file structure:

```
└── callbacks/        # Callbacks for evaluation and logging during training.
└── pipelines/        # Pipelines for batching and preprocessing data.
    └── multistage_pipeline/
    └── sample_processors/  
└── configs/          # YAML files for configuring experiments, models, datasets, etc., using Hydra.
    └── callbacks/
    └── pipelines/
    └── datasets/
    └── experiments/
        └── shapenet/
    └── model/
    └── optim/
    └── tracker/
    └── trainer/
    train_shapenet.yaml
└── datasets/         # Dataset classes for loading data.    
└── jobs/             # SLURM job scripts to run experiments.
└── model/            # Model definitions.
└── modules/          # Individual modules needed for model components.
└── notebooks/        # Jupyter notebooks.
└── scripts/          # Stand-alone scripts, e.g., to collect dataset statistics or generate visualizations.
└── trainers/         # Trainer classes that manage the training loop.
```

The minimal required folder structure for any `ksuit` project is:

```
└── callbacks/ 
└── pipelines/   
└── configs/  
└── datasets/ 
└── model/
└── trainers/   
```

The [`configs/`](./configs) folder generally mirrors the same folder structure as the root, meaning that for all the modules we define, we also define base configurations.
Every `ksuit` project requires the following modules to run (in alphabetical order):

  1.  **Callbacks**: Classes used to compute metrics and statistics during training at specific moments that are specific for the project (can be empty though).
  2.  **Configs**: The configurations for all individual modules and the entire training pipeline.
  3.  **Pipelines**: Defines how to collate and preprocess loaded data into a batch. 
  The pipelines act as the interface between the `Dataset` and the `Trainer`.
  4.  **Dataset**: The interface between the data stored on disk and the collator. This class defines how to load individual tensors per sample/data point.
  5.  **Trainer**: Given a model and a batch of data created by the collator, the `Trainer` class runs the model's forward pass and computes loss values.
  6.  **Model**: Defines the model's architecture and its forward pass.

**Note:** We do not need config files for the files located in [`scripts`](./scripts/), [`notebooks`](./notebooks/), [`modules`](./modules/), and [`jobs`](./jobs/) folders.

### Setup

Create a virtual environment in the root of the [`core`](../) repository, with uv with the python version specified in pyproject.toml (3.12) and install dependencies:
```bash
  uv venv --python 3.12
  source .venv/bin/activate
  uv sync
```

#### Project boilerplate code

In the folder [`boilerplate_project/`](../boilerplate_project), we provide a minimal training loop on a toy task, which provides the minimal components needed for a train run in `ksuit`.
Next to this tutorial, you can also have a look at that project folder, to see how the minimal code of each module looks like, required for a `ksuit` training.

## Tutorial 

Now we start with the tutorial.
We go over each module in logical order; however, we have to jump back and forth between some modules every now and then since they are involved in multiple aspects of the entire pipeline.

### Configutation ([`configs/`](./configs/))
-----
At the core of all experiments are the [configuration files](./configs).
All experiments are driven by **YAML configuration files**. 
The main entry point for training is a file like [**`train_shapenet.yaml`**](./configs/train_shapenet.yaml). 
This file uses **Hydra** to compose a complete configuration from smaller, modular files located in subdirectories, and defines how to run an experiment for the **ShapeNet-Car dataset**.

As said, [**`train_shapenet.yaml`**](./configs/train_shapenet.yaml) defines how to configure an experiment for the ShapeNet-Car dataset.
The file is given below:

```yaml
# @package _global_

# Define key values here that are used multiple times in the config files.
dataset_root:  /nfs-gpu/simulation/data/shapenet_car_v3
collator_kind: tutorial.datasets.ShapeNetCarDataset

defaults:
  - model: ??? # models are undefined and will be defined per experiment 
  - trainer: shapenet_trainer
  - datasets: shapenet_dataset
  - tracker: ??? # trackers are undefined and will be defined depending on either development, training or evaluation
  - callbacks: training_callbacks
  - collator: shapenet_pipeline
  - optim: adamw
  - _self_

```

In this file, we define the base configs for all modules/classes we have to instantiate.

To run an experiment with `ksuit`, you need to define and instantiate the configuration for: 
  1. The **model** to train.
  2. The **trainer** to use.
  3. The **training callbacks**. The callbacks are part of the `Trainer config`, but we already define them in the main config file.
  4. The **(WandB) tracker** to use. 
  5. The **dataset** to use. 
  6. The **collator**.  The collator config is part of the `Dataset config`, but we already defined it in the main config file.
  7. The **optimizer** (optim). The optimizer is defined in the model config, but we already defined it in the main config file.. 
  Next to that, we also define keys that we want to use in other base configs, and are used multiple times, such as the `dataset_root` and `collator_kind`.

Most of these modules/class configs do not change when you train a different model on the same dataset. 
For instance, when training a model on ShapeNet-Car, everything except the model (i.e., which model to train) and tracker (i.e., which WandB project to use) does not change.
Therefore, we use **default configurations** to ensure consistent hyperparameters for all experiments we run for a dataset.
For example, the **[`configs/datasets/shapenet_dataset.yaml`](./configs/datasets/shapenet_dataset.yaml)** defines the configuration for the ShapeNet-Car dataset class. 
To use the ShapeNet-Car `dataset/config` file in an experiment, you must specify `datasets: shapenet_dataset` in the `train_shapenet.yaml` file.

#### Running specific experiments

To run a specific experiment, you need to define an **experiment YAML file**. 
In this file, you define: (1) a **model**, (2) a **tracker**, and (3) override any configuration variables specific to that experiment.

For example, the experiment YAML for UPT ([`configs/experiments/shapenet/upt.yaml`](./configs/experiment/shapenet/upt.yaml)) looks like this:

```yaml
# @package _global_
defaults:
  - override /model: upt # this default was undefined 
  - override /tracker: shapenet_experimental # this default was undefined
  - override /optim: lion # we override the default AdamW optimizer for UPT from AdamW to LION.

name: shapenet-car-up # give a different name to this experiment 

# all the hyper-parameters below are overrides from the defaults, depending on specific experiments.
trainer:
  precision: float16 # instead of using float32, we use float16

collator:
  num_supernodes: 3586 # num_supernodes is None by default, since not all models use supernodes
  num_surface_queries: 3586 # num_surface_queries is None by default, since not all models use queries.
  num_volume_queries: 4096  # num_volume_queries is None by default, since not all models use queries.
```

In this example, we select **`upt.yaml`** from the `configs/model` folder and the **`shapenet_experimental.yaml`** tracker from the `configs/tracker` folder. 
We also override the trainer's precision from `float32` to `float16` and change some variables in the collator.

Next, we only have to use the following command:

```bash
uv run noether-train --hp configs/train_shapenet.yaml +experiment/shapenet=upt
```

This commands runs ```noether-train``` from the ```noether``` CLI, with the ```--hp configs/train_shapenet.yaml``` hyper-parameters, using the overrides from the  ```+experiment/shapenet=upt``` experiment. 
Please have a look at all the base configs to see what is defined where and how the files are composed.

Some of the config files make use of references to other base configs, instead of hardcoding everything.
For example, the dataset config file can look like this:

```yaml
train:
  root: /nfs-gpu/simulation/data/shapenet_car_v3
  kind: tutorial.datasets.ShapeNetCarDataset
  split: train
  collator: ${collator}
```

The `collator` key does not take a string as value, but a reference to the collator key that is already defined in `configs/train_shapenet.yaml`.
By doing this, we keep our configuration:
1. **DRY (Don't Repeat Yourself):** It avoids duplicating the collator definition for both train and test datasets (since we use the same collator in this example), making your configuration more concise and easier to maintain.
2. It's a cornerstone of **modular configuration systems**, allowing you to build complex configurations from smaller, reusable components and allowing for overrides.

#### Modifying hyperparameters

If you want to try certain variants of the same experiment, you can simply override or add hyperparameters via either the **CLI** or by creating a **new experiment YAML file** with a different name. 
For example, if you want to change the hidden dimension of UPT from `192` to `256` and the number of attention heads from `3` to `4` (since `hidden_dim % num_heads == 0`), simply run the following command:

```bash
uv run noether-train --hp configs/train_shapenet.yaml +experiment/shapenet=upt model.hidden_dim=256 model.num_heads=4
```

The name of the WandB run will be addressed accordingly. 
The other option would be to create a new experiment file, for example ```upt_dim_256.yaml```, where you override ```model.hidden_dim```, ```model.num_heads```, and change the name.
We will discuss later how to launch jobs and run multiple experiments.

As said, it is recommended to **avoid unnecessary duplication** (keep it **DRY**). 
For example, if you want to try out models with different hyperparameters, do not create multiple copies of the same model file with only one hyperparameter changed (for example, the same model with different hidden dims). 

### The Dataset ([`datasets/`](./datasets/))
-----
The first class we will define is the `Dataset`. 
This class acts as a bridge between the raw (or pre-processed) data (stored on disk or other storage) and the **collator**, which collates data samples into a batch for the model.
Our custom [ShapeNetCarDataset](./datasets/shaptenet_dataset.py) dataset class extends `ksuit.data.Dataset`, which in turn extends the standard `torch.utils.data.Dataset`.

#### The `getitem_*` pattern

In a standard PyTorch `Dataset`, you typically define a single `__getitem__` method. 
This method receives an index (`idx`) and is responsible for loading all the necessary data (mostly tensors) for that data point. 
A drawback to this approach is that `__getitem__` can become overly complex if you need to load different combinations of data for different configurations (e.g., training vs. validation, or when different models rely on different input for the same dataset).

The `ksuit.data.Dataset` uses a more modular approach. 
Instead of one monolithic `__getitem__` for all tensors that belong to a data point, you define a separate `getitem_*` method for each individual tensor that you might want to load from disk.

Here are two examples:

```python
def _load(self, idx: int, filename: str) -> torch.Tensor:
    """
    Loads a tensor from a file within a specific sample directory.

    Args:
        idx: Index of the sample to load.
        filename: Name of the file to load from the sample directory.

    Returns:
        The loaded tensor.
    """
    # Use modulo to handle dataset repetitions
    idx = idx % len(self.uris)
    sample_uri = self.uris[idx] / filename
    return torch.load(sample_uri, weights_only=True)

def getitem_surface_position(self, idx: int) -> torch.Tensor:
    """Retrieves surface position (num_surface_points, 3)."""
    return self._load(idx=idx, filename="surface_points.pt")

def getitem_surface_pressure(self, idx: int) -> torch.Tensor:
    """Retrieves surface pressure (num_surface_points, 1)."""
    return self._load(idx=idx, filename="surface_pressure.pt").unsqueeze(1)
```

Each `getitem_*` method is responsible for loading one specific tensor. 
As shown, you can use helper methods like `_load` to keep your code clean and avoid repetition (Don't Repeat Yourself - **DRY**). 
Like the standard `__getitem__`, these methods take an index to identify which sample to load.

#### Available data in ShapeNet-Car

For our ShapeNet-Car example, each data point consists of several PyTorch tensors stored on disk (we already preprocessed the raw data into PyTorch Tensors). 
For each available tensor, we define a corresponding `getitem_*` method.

We have the following data available (with matching `getitem_` methods):

  * **Surface Position**: Coordinates of the surface mesh points.
  * **Surface Pressure**: Pressure value of each point on the surface mesh. 
  * **Surface Normals**: Normal vector of each point on the surface mesh.
  * **Volume Position**: Coordinates of the points in the surrounding volume mesh.
  * **Volume Velocity**: Velocity vector at each point in the volume mesh.
  * **Volume Normals**: Normal vector of each volume point, directed towards the nearest surface point.
  * **Volume SDF**: Signed Distance Function (SDF) value for each volume point to the nearest surface point.

Note that there is no `getitem_surface_sdf` method. 
The SDF for the surface mesh itself would be a tensor of all zeros, so we don't need to load it.

For some of the models we use in this tutorial, we need additional `getitem` methods.
For example, for GINO, we need a regular grid of points. 
Since this grid is the same for all data samples, we generate it once within the `Dataset` class via `getitem_grid_pos` (however, another option would be to do this in the collator).

#### Dataset configuration

To select which data tensors to load for a given (trainer) run, you can use `included_properties` and `excluded_properties` in the dataset config
The collator will use this set of keys to call the corresponding `getitem_*` methods.

For example to avoid `volume_sdf` from being loaded:
```yaml
train:
	kind: emmi.dataset.Shapenet
  # ... 
  excluded_properties:
  - volume_sdf
  # ...
```

The `properties` property dictates which `getitem_*` methods will be called by the data loader. 
The example above would load all available data tensors for ShapeNet-Car.


#### Key methods of the Dataset class

Beyond the `getitem_*` methods, a few other methods are important in the dataset class.

 `__len__`

The `__len__` method defines the total size of the dataset for one epoch. 
It's calculated by multiplying the number of unique data samples (URIs) by `num_repeats`. 
Repeating the dataset is useful during evaluation to reduce variance by iterating over the same samples multiple times.

```python
def __len__(self):
  """
  Returns the length of the dataset.
  """
  return len(self.uris) * self.num_repeats
```

`get_normalization_stats`

This static method returns a dataclass containing all the normalization constants (e.g., mean, std) for this specific dataset. 
The collator uses these statistics to normalize the input data before passing it to the model.

```python
@staticmethod
def get_normalization_stats() -> ShapeNetCarStats:
  """Returns the normalization stats for the dataset."""
  return ShapeNetCarStats()
```

`__init__`

The `__init__` method handles the initial setup. 
This includes finding all data URIs on disk and assigning them to indices so we can iterate over the dataset. 
This logic is often dataset-specific.
The remaining methods in the class are similar to a standard PyTorch Dataset and are therefore beyond this tutorial's scope. 
If you're unfamiliar with PyTorch Dataset classes, we recommend reviewing their [documentation](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) 

### The  Multi-Stage Pipeline ([`pipelines/`](./pipelines/))
-----

The pipeline serves as an interface between the dataset and the model/trainer. 
It defines how to combine individual samples from the dataset into a batch that's fed to the model. 
This batch contains the model inputs for the forward pass and the corresponding targets needed to compute the loss.

The multi-stage pipeline has three sequential stages:

1.  **Sample processing**: Sample processors act on individual data samples (i.e., data points).
2.  **Collation**: The collator combines the individual samples into a batch.
3.  **Batch processing**: Batch processors act on the entire batch.

This sequence gives the **multi-stage pipeline** its name. 
In this project, most of the multi-stage pipelines's computation occurs during the pre-processing stage.

A basic implementation of a custom `MultiStagePipeline` looks like this:

```python
from ksuit.data.pipeline import MultiStagePipeline

class CustomMultiStagePipeline(MultiStagePipeline):
  def __init__(self, **kwargs):
    super().__init__(
      sample_processors=[],
      collators=[],
      batch_processors=[],
      **kwargs,
    )
```

You need to provide three lists to the multi-stage pipeline: one for sample processors, one for collators, and one for batch processors. 
The `MultiStagePipeline` iterates through each list sequentially. 
The output from one sample processor becomes the input for the next, making the order of operations **crucial** for all three stages.

> 💬 **Personal note -- Maurits**:
> I've changed my mind a million times about the collator implementation, and I'm still unsure of the ideal approach. 
As mentioned earlier, the pipeline serves as the interface between the dataset and a model. 
With nine distinct models and three datasets (for the AB-UPT paper), we could, in theory, end up with 9×3=27 different pipelines for this project.
>
> However, because the interfaces for most models and datasets are similar, pipelines can often be reused with minor adjustments. 
Another option is to design a single pipeline for each dataset that serves all models. 
The challenge with this, though, is that such a pipeline quickly becomes a "god class" – excessively complex with intricate logic.
>
> I believe the solution lies somewhere in the middle, but I haven't pinpointed it yet. 
For now, I've opted for the "one-pipeline-for-all-models" approach, which I'll explain further. 
I realize this isn't an ideal solution, but I was less satisfied with the alternatives I explored. 
I'd be happy to discuss your thoughts on the best possible solution.

#### Sample Processors

To understand the [ShapenetMultistagePipeline](./pipelines/multistage_pipelines/shapenet/shapenet_multistage_pipeline.py), it's essential to understand the data processing flow for this project.

We're dealing with CFD aerodynamic simulations that have both a surface and a volume mesh/field. 
Each point in these fields has three coordinates `(x, y, z)`, one or more target values (e.g., `pressure`, `velocity`, etc.), and potentially additional features (e.g., `SDF`, surface/volume normals). 
The target values and features can vary depending on whether the point belongs to the surface or volume and which dataset is used. 
From now on, we'll refer to these additional features as **physics features**. 
We do not consider global features for this project.

For the AB-UPT paper, we decided to use only coordinates as model inputs. 
However, models like Transolver are **designed** to also handle these additional **physics features** for each point. 
Therefore, I've built all the models to be capable of including these additional **physics features**. 
The downside is that this adds complexity to the collator, but I believe building a pipeline that can handle these features is good practice.

**NOTE:** As mentioned, we can only handle additional features per mesh point (on the surface or in the volume) if the feature dimensionality is the same for both surface and volume points. 
Global features are not supported at the moment.

The models we use (except for AB-UPT) can be roughly **divided** into two classes:

1.  **Point-based models**, where the input points to the model's encoder are also the points used for predicting the output values (e.g., `PointNet`, `Transolver`, `Graph-UNet`).
2.  **Query-based models**, which use additional query points (distinct from the input points) for predicting output values (e.g., `UPT`, `LNO`, `OFormer`, `GINO`).

This means we have to build a sample processor pipeline that works for both point-based and query-based models.

We will now outline the pipeline required for these models:

1.  Some input tensors have **constant** values, for example, the `SDF` for the surface mesh is always zero. 
Therefore, we first create **default tensors**. 
Because this step occurs before batch collation, it's considered a **pre-processing** step.
The regular grid required by GINO could also be treated as a default tensor, but for convenience, I've made it part of the dataset itself.
2.  Since we often use only input coordinates, the models are **unaware** of which points belong to the surface mesh and which belong to the volume mesh. 
Therefore, we create a **surface mask**, which indicates which points are on the surface (its negation is the **volume mask**).
 When sampling query points, we need to create a surface mask for the query points as well.
3.  Most tensors we retrieve from the dataset need to be normalized. 
The next step in the sample processor pipeline is to normalize these various tensors.
4.  Next, we normalize the input coordinates (both on the surface and in the volume). 
For clarity, we made this an individual step, but it could have been combined with step (3).
5.  Next, we subsample the entire simulation mesh to a specified number of input points and, if used, query points. 
For both input and query points, we define how many to sample from the surface and how many from the volume. If we train AB-UPT, we sample anchor points instead of input/query points.
6.  Most models take the surface and volume points concatenated as input (this also applies to query points). 
Therefore, we need to concatenate the surface **and** volume points into a single tensor. 
This concatenation is also required for the output targets and, if used, the input's physics features.
7.  If we use query points, their corresponding physical quantities become the model's prediction targets. 
If we only use input points, their values are the output targets (labels). 
Hence, we need to rename the relevant values to `targets` based on whether the model uses input points or query points for its predictions.
8.  Some models, such as `GINO`, need model-specific pre-collating. Therefore, the final stage consists of model-specific pre-collaters.

The high-level pipeline is visualized in the image below.

![Alt text](./images/shapenet_car_models_data_flow.png)

This entire pipeline is implemented in the `_build_sample_processors_pipeline` method in the `ShapenetMultistagePipeline` class, which composes the list of sample processors based on the 8 steps listed above. 
This method returns a list of individual `ksuit.data.SampleProcessor` instances. 
Note that the order is important, as the sample processors are called sequentially.

When the multi-stage collator runs, the sample processors are called as follows:

```python
# pre-process on a sample level
for sample_processor in self.sample_processors:
  samples = sample_processor(samples)
```

Each **sample processor** takes a list of samples as input and returns the processed list. 
As mentioned, the order is **crucial**. 
Each `sample_processor` must implement the `__call__(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]` method. 
This method receives a list of samples as input. Each sample is a dictionary, and all dictionaries in the list share the same keys. 
The `sample_processors`'s goal is to apply a specific processing step to the corresponding values for one or more keys across all sample dictionaries.

#### Denormalization

Some sample processors apply a normalization step, which we sometimes have to reverse for evaluation purposes. 
You can call `sample_processor.inverse(batch)` to denormalize a batch of data; however, you have to be very **careful** with this. 
Below is the code for the `denormalize` method in the `MultiStagePipeline` class.

```python
def denormalize(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  """Inverts the normalization of all items in the whole batch by calling `denormalize` of sample/batch processors

  Args:
      batch: Collated batch with normalized values

  Returns:
      Collated batch with unnormalized values.
  """
  # denormalize each item separately because multiple items could be mapped back to the same origin item
  denormalized_batch = {}
  for key, value in batch.items():
      og_key = key
      for sample_processor in reversed(self.sample_processors):
          key, value = sample_processor.denormalize(key=key, value=value)
      denormalized_batch[og_key] = value
  return denormalized_batch
```

**Note:** The batch that serves as input for `denormalize` can be any batch of data.

As you can see, we loop over the sample processors in **reverse** order and call the `denormalize` method on each one.
A sample processor should only denormalize a value from the batch dictionary if the **key** matches one it originally processed and if the operation is reversible.

For example, the `denormalize` function of the `DefaultTensorSampleProcessor` looks as follows:

```python
def denormalize(self, key, value) -> tuple[str, torch.Tensor]:
    """ Does not contribute to denormalization -> identity function"""
    if key == self.item_key_name:
        raise ValueError("Default tensor cannot be denormalized.")
    return key, value
```

If a key provided to `DefaultTensorSampleProcessor.denormalize` doesn't match the key of the default tensor it created (`self.item_key_name`), the method does nothing and returns the original input. 
However, if `item_key_name` is presented to the `denormalize` method, we must throw an error, since this step cannot be denormalized.

As another example, the `denormalize` method of a `MomentNormalizerSampleProcessor` looks like this:

```python
def denormalize(self, key: str, value: torch.Tensor) -> tuple[str, torch.Tensor]:
  """Inverts the normalization from the __call__ method of a single item in the batch.

  Args:
      key: The name of the item.
      value: The value of the item.

  Returns:
      (key, value): The same name and the denormalized value.
  """
  if key not in self.items:
      return key, value
  assert value.ndim == self.mean_tensor.ndim
  denormalized_value = value * self.std_tensor.to(value.device) + self.mean_tensor.to(value.device)
  return key, denormalized_value
```

Denormalization is only applied if the key from the batch matches a key that this pre-processor originally normalized (i.e., is in `self.items`). 
Otherwise, the inputs are returned unchanged.

Instead of using `pipeline.denormalize(batch)`, which calls the `denormalize` method for the entire pipeline of post and sample processor you can also retrieve a specific processor and only call the individual `denormalize` method of that collator. 

To retrieve a specific sample processor from the collator, you can use the `get_sample_processor` method from the MultiStagePipeline class.
It  takes a predicte function as input, that returns true if the correct sample processor is given as input to the predicte.
From example, to find the normalizer sample processors that normalize the `surface pressure` values, use the following predicate 
```python
lambda p: isinstance(p, MomentNormalizationSampleProcessor) and p.item == 'surface_pressure'
```
. 
This will return `true` only for a `MomentNormalizationSampleProcessor` sample processor, and when the `item` (i.e., key) this normalizer works on is `surface_pressure`.

I would now recommend reviewing the `ShapenetMultistagePipeline` class to see how the sample processor pipeline is constructed and how the individual sample processors work.

The reason that it is hard (in my opinion) to split this pipeline into multiple classes is the order in which you have to call those sample processors.
It is not trivial to split the code into two parts (for example, point sampling and query sampling), because some of those depend on each other. 

#### Collators

The code for calling the collators in the multistage pipeline looks as follows:

```python
batch = {}
for batch_collator in self.collators:
  sub_batch = batch_collator(samples)
  # make sure that there is no overlap between collators
  for key, value in sub_batch.items():
    assert key not in batch, f"Key '{key}' already exists in batch. Collators must not overlap in keys."
    batch[key] = value
```

Each collator defines how to merge certain keys in each sample into a batch.
In most cases, `emmi.pipeline.collators.DefaultCollator`, where the tensors are simply concatenated along the batch dimension, will do the job.
However, sometimes, when creating sparse tensors, for example, a more sophisticated way of collating is required. 
We define the collator pipeline in the `_build_collator_pipeline` method.
Only when we are dealing with a supernode, we require additional collator classes.

#### Batch Processors

In this project, we do not use any batch processors.
Nevertheless, they work in the same way as the sample processors. 
However, instead of a list of samples, they get the collated batch of samples as input.

### The trainer (`trainers/`)
-----

The [`AutomotiveAerodynamicsCFDTrainer`](./trainers/automotive_aerodynamics_cfd_trainer.py) (i.e., the Trainer class) is a specialized trainer designed for automotive Computational Fluid Dynamics (CFD) tasks, particularly for the datasets AhmedML, DrivAerML, and ShapeNet-Car. 
Its primary role is to compute the training step by processing model outputs, computing a flexible weighted loss, and returning the results.

#### Initialization and Configuration (`__init__`)

The most important variables in the ```__init___```are the loss weights, which give you fine-grained control over the training objective.

The loss system has two levels of weights:

  * **Individual Weights**: Parameters like `surface_pressure_weight` and `volume_velocity_weight` control the importance of a specific physical quantity in the total loss.
  * **Group Weights**: The `surface_weight` and `volume_weight` parameters apply an additional weight to all surface-related or volume-related losses, respectively.

During initialization, the trainer uses these weights to build an internal `loss_items` list. 
The `output_modes` parameter (e.g., `['surface_pressure', 'volume_velocity']`) tells the trainer which of these potential losses should be computed during training.

#### The Training Step (`forward` method)

The trainer's main method is the `forward` method. 
Unlike the `forward` method of a `torch.nn.Module`, this one defines a single training step. 
It's quite straightforward and performs three key actions:

1.  **Get Model Predictions**: It passes the entire data `batch` to the model to get the predictions.
    ```python
    output = model(batch)
    ```
2.  **Compute Loss**: It uses its `loss_compute` method to calculate the various losses based on the model's `output` and the ground truth **targets** from the `batch`.
3.  **Return Results**: It returns a `TrainerResult` dataclass containing the final loss information.

The `TrainerResult` data class holds two essential attributes:

  * `total_loss`: A single `torch.Tensor` representing the final, combined loss for the training batch. It's the sum of all computed individual (weighted) losses.
  * `losses_to_log`: A dictionary containing the individual, named losses (e.g., `{'surface_pressure_loss': tensor(...)}`). This is useful for detailed logging and monitoring with tools like TensorBoard.

#### Custom Loss Calculation (`loss_compute`)

This method contains the core logic of the trainer, computing the loss. 
It calculates the final loss by iterating through the `loss_items` that were configured during initialization.
For each item (like `surface_pressure`), it first checks that its weight is non-zero and that the model produced a corresponding output key.
This flexible system allows you to easily experiment with different combinations of output objectives without changing the underlying code.

### Models (`models/`)
-----

Building models in `ksuit` is relatively straightforward and quite similar to plain PyTorch models that inherit from `torch.nn.Module`.

To be compatible with the `ksuit` `Trainer`, all models must inherit from `ksuit.models.Model` (or `CompositeModel`, but we don't use that for this project). 
Beyond this requirement, a model is implemented just like a standard PyTorch model by defining its layers in the constructor and implementing the `forward` method. 
A `Model` receives its parts of the input arguments from the `Trainer` via the `get_model_instantiation_kwargs` method, allowing the model to be configured with parameters provided by the `Trainer`.

#### The `BaseModel` for standardization of all models

To unify the input representation, output structure, and input conditioning (i.e., distinguishing whether coordinates belong to the volume or a surface mesh) across all baseline models, a `BaseModel` class is provided. 
It contains common utilities that can be used by any model, including:

1.  An MLP projection for surface and volume bias.
2.  A linear projection to map physics features to the model's hidden dimension.
3.  A sine-cosine positional embedding layer for input coordinates.
4.  An output projection layer from the hidden dimension to the number of predicted values.

#### Handling model outputs

While most models use a single output layer for all input points, the physical quantities predicted for surface points often differ from those for volume points. 
The `gather_outputs` method in the  `BaseModel` class is designed to handle this. 
It takes the entire output tensor and a surface mask, then splits the output tensor to correctly isolate the predictions for surface points from those within the volume.
Using the `gather_outputs` method for all models, the output dict is structured in such a way that the `loss_compute` method in the trainer can compute the losses in the correct way.
By doing this, we can use the same trainer for all methods.

For example, the outputs might be structured as follows:

  * `surface pressure` is assumed to be at dimension 0.
  * `volume velocity` is at dimensions 1:4, etc.

#### Managing model-specific inputs

Not every model requires the same set of inputs for its `forward` method. 
To manage this cleanly, a decorator is used to filter the data batch, ensuring that only the required fields are passed to the model's `forward` call.

Therefore, each model's YAML configuration file must specify its **`required_batch_modes`**. 
These are the keys that the `validate_required_batch_input` decorator uses to select the appropriate data from the batch. 
For example, the `required_batch_modes` for a `PointNet` model would look like this:

```yaml
required_batch_modes:
  - input_position
  - surface_mask_input
```

> [\!NOTE]
> The `validate_required_batch_input` decorator is a convenience feature, not a strict requirement of the `Model` class. It was implemented to simplify development by allowing the data collator to return the batch with all possible data. 
The decorator then filters this batch for each specific model, preventing errors by ensuring the model only receives the inputs it expects.

### Callbacks (```callbacks/```)
-----

A **callback** is an object (in our framework, but it can also be a function) that can perform actions at various stages of the training loop, such as at the beginning or end of training, an epoch, or an update step.

The `SurfaceVolumeEvaluationMetricsCallback` is a specific callback that runs the current model on a separate validation or test set, computes error metrics, and logs them. 
This class inherits from `PeriodicCallback`, meaning its main logic is executed at regular intervals. 
It requires a `dataset_key` (e.g., "validation" or "test") to identify which dataset to use for evaluation. 
It also takes a reference to that dataset's **collator**, which is crucial for denormalizing data.

The `_periodic_callback(...)` method is the main entry point called by the trainer. 
It  works as follows:
1.  It calls `iterate_over_dataset`, a method that loops through the entire evaluation dataset.
2.  For each batch, it executes the `_forward` method to calculate the metrics for that sample. A batch in the context of collator usually contains one sample/data point, since most metrics are computed per sample. However, you can define the batch size for the callback in the config files. The `_forward` method should return a dict with the metrics for the sample that is processed by the method. The `iterate_over_dataset` will return a collated dict with all the metrics. 
3.  After collecting metrics for the whole dataset, it logs them using a writer (e.g., to Weights & Biases) and prints them to the command line.

The `_forward` method defines the evaluation logic for  (usually) a single batch of data.
In `tutorial/configs/callbacks/training_callbacks.yaml`, we define the list of callbacks we want to use for the trainer class. 
To define how often a periodic callback (i.e., `_periodic_callback`) should be triggered, set one of the following arguments in your configuration:
* `every_n_epochs`: Triggers the callback every N epochs.
* `every_n_updates`: Triggers the callback every N model update steps.
* `every_n_samples`: Triggers the callback after every N samples have been processed.
You cannot define multiple of those arguments. Next to the intercal, we can also define the `batch_size`, which is usually set to `1` to compute metrics per sample. 

Models are typically trained on **normalized data** (e.g., with zero mean and unit variance) for numerical stability. 
However, to calculate physically meaningful errors, we must reverse this normalization. 
The `_forward` method uses the collator to **denormalize** both the model's predictions and the ground truth targets back to their original physical units.

Within the `_denormalize` method, the collator is used to find the correct sample processor and convert tensors back to their original scale before they are used to compute the final, unnormalized metrics.

For each output, we calculate the following metrics:
1.  **Mean Squared Error (MSE):** The average of the squared differences between the prediction and the target.
2.  **Mean Absolute Error (MAE):** The average of the absolute differences between the prediction and the target.
3.  **Relative L2 Error:** The Euclidean norm of the error vector divided by the norm of the target vector. This measures the error relative to the magnitude of the ground truth.

Some callbacks need to track information on every single training step, even if they only perform their main action periodically (e.g., after an epoch). 
A good example is calculating an **exponential moving average (EMA)** of the model's weights. 
To implement a callback like this, you can use the `_track_after_accumulation_step` or `_track_after_update_step` methods to implement the track step every other step.
The `_periodic_callback` is still used as defined.

### Development and debugging

The notion of debugging is a bit ill-defined in deep learning settings. 
If your code does not crash, it does not imply that the model you train is meaningful.
There are two recommended ways of debudding.
The first way is related to making sure your training/evaluation runs, and that the pipeline is correct.

When working in `VSCode`, use the following example to debug a model

**NOTE**: This is an example from my IDE (Maurits), of course, you need to set the correct file paths for your own environment.

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Development ShapeNet PointNet",
            "type": "debugpy",
            "request": "launch",
            "env": {
                "WANDB_DISABLED": "true"
            },
            "program": "/home/mbleeker/projects/emmi.ai/core/src/noether/training/cli/main_train.py",
            "args": [
                "--hp",
                "/home/mbleeker/projects/emmi.ai/core/tutorial/configs/train_shapenet.yaml",
                "+experiment/shapenet=pointnet",
                "tracker=disabled",
                "+seed=1"
            ],
            "console": "integratedTerminal",
            "justMyCode": false
        },
    ]
}
```

Now the training pipeline for a PointNet model can be run in debug mode, and break points can be put at any point inside either the ksuit package or inside the code of the tutorial itself.
By doing this, the dimensionality of tensors can be inspected, certain values can be checked, etc. 
Select the same virtual environment as you use for training. If you work with remote development in your IDE, the program automatically runs on a GPU, which makes the forward/backward pass super fast. 
We disable wandb for debugging runs, to prevent an overflow of short training runs in our WandD dashboard.

Even though a training loop does not crash, it does not mean that the model is trained properly. 
To debug model training, you sometimes have to try certain things and evaluate after an entire training run to verify that a new model/trainer setup works.
To not overwhelm the WandB with experimental/debugging runs, we recommend using a different WandB tracker than for the 'official experiments'.

### Running jobs

To run all the baseline models + AB-UPT in one go on SLURM, simply run

```
sbatch train_shapenet.job
```

in the `tutorial/jobs` folder.

In the folder `tutorial/jobs/experiments`, we define a job array (i.e., an array with different experiments/jobs) for all the experiments we want to run in `shapenet_experiment.txt`.

You could add extra rows with different seeds, or experiment variants, however you like to this `*.txt` file. 

The flag `#SBATCH --array=...` defines how to run the job array. 

For example, ```#SBATCH --array=1-10```, will run row 1 to 10, from ```/jobs/experiments/shapenet_experiments.txt```.
 ```#SBATCH --array=1,5,9``` will run row, 1, 5, and 9.
 ```#SBATCH --array=1-10%5``` will run row 1 to 10, but with an upper-bound of max 5 running jobs at the same time. If one of the 5 jobs is finished, the next job in the array (i.e., row 6) will start. This is especially useful with large job arrays, and you do not want to occupy the entire cluster. 

### Running a single experiment 

To run a single experiment, run the following line on the command line:
```bash
uv run noether-train --hp {user path to core}/core/tutorial/configs/train_shapenet.yaml. +experiment/shapenet=pointnet tracker=disabled +seed=1"
 ```

### Distributed Training
Single-node multi GPU training is supported both via direct execution and when running through SLURM. 
When running in SLURM make sure to set the `--gpus-per-node` and `ntasks-per-node` to the number of GPUs used. 
This will start one process per GPU.

If running outside of SLURM use `uv run noether-train` like above and this will spawn one process for every GPU that is available on the system and set visible with `CUDA_VISIBLE_DEVICES`. 

To run multi-node training outside of SLURM, start one process per GPU and make sure to set `MASTER_ADDR` and `MASTER_PORT` environment variables accordingly.
