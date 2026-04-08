# Conformity LLMs: Image-Based Task Pipeline

This repository generates synthetic images for three perceptual tasks and runs LLM-based experiments to evaluate performance and conformity patterns. The workflow is notebook-driven and split into two experiment paths (standard and Ovis) because they require different imports and dependencies.

## System Requirements

### Tested software environment

- Operating system tested: `Ubuntu 22.04.5 LTS` (Jammy)
- Python tested locally in this repository: `Python 3.10.12`
- The notebook execution logs in this repository also show runs on Python `3.11` environments (e.g., `/opt/conda/lib/python3.11/...`)

### Python dependencies

This repository now provides three installable requirement files:

- `requirements.txt`
- `requirements-general.txt`
- `requirements-ovis.txt`

Recommended for full reproducibility (single environment that runs both paths):

```bash
python3 -m pip install -r requirements.txt
```

This unified file is pinned to a compatibility intersection, because notebook constraints differ:

- Standard/general notebooks use `transformers>=4.49`
- Ovis notebooks use `transformers<4.54.0`

So `requirements.txt` pins `transformers==4.53.3` (compatible with both).

If you want separate environments:

```bash
# General (non-Ovis) path
python3 -m pip install -r requirements-general.txt

# Ovis path
python3 -m pip install -r requirements-ovis.txt
```

Observed `transformers` versions in notebook outputs:

- General/non-Ovis notebooks: `4.51.3`, `4.53.1`, `4.53.3`, `4.55.2`
- Ovis notebooks: `4.53.3` (consistent with `<4.54.0`)

### Model versions / model IDs

All model IDs used by the notebooks and `simulations_core.py` are listed in:

- `MODEL_VERSIONS.txt`

This file maps each `model_label` (used in `run_all.ipynb` and general notebooks) to the exact Hugging Face model ID loaded in code.

### Non-standard hardware requirements

- A CUDA-capable GPU is required for practical execution.
- `simulations_core.py` uses GPU-oriented settings (`device_map="auto"`, `.cuda()`, `torch_dtype=torch.bfloat16`, and optional bitsandbytes quantization).
- Larger models (e.g., 16B/24B/27B/32B/34B/72B variants) may require high VRAM and/or multi-GPU setups.

### Reproducibility note

Exact reproducibility can vary with:

- GPU architecture and CUDA stack
- PyTorch / Transformers / bitsandbytes versions
- Model-side updates on remote hubs

For this reason, use the pinned `requirements.txt` and the exact model IDs in `MODEL_VERSIONS.txt`.

## Repository Structure

- `create_images/`  
  Notebooks to create images for each task and model. File naming follows:  
  `create_images_<task>_<model>.ipynb`
- `images/`  
  Output directory for all generated images. Each task/config/model gets its own subfolder.
- `general_notebook.ipynb`  
  Main experiment notebook (standard dependencies).
- `general_notebook_ovis.ipynb`  
  Main experiment notebook for the Ovis setup (different imports/dependencies).
- `run_all.ipynb`  
  Orchestrates running the general notebooks across tasks/models.
- `data/`  
  Output directory for experiment results (pickles, csvs, etc.).
- `simulations_core.py`  
  Shared core logic used by the general notebooks.
- `LICENSE.txt`  
  MIT license file for code and documentation reuse.

## Tasks, Configurations, and Models

The project includes three tasks:

- `asch_lines`
- `color_recognition`
- `dots_estimation`

For each task, images are created for:

- General image pools
- Perplexity-based difficulty/ambiguity configurations

Each of these is generated per model (e.g., `qwen`, `qwen2`, `mistral`, `gemma`, `ovis`), depending on the notebook you run. The exact model list is defined inside the notebooks.

## End-to-End Pipeline (Required Order)

1) **Generate images (all tasks, all configurations, all models)**  
   Use the notebooks in `create_images/`. Each notebook creates images for a single task/model pair and saves them under `images/`.

   Output layout example:
   - `images/images_lines_<model_label>/`
   - `images/images_lines_perplexity_diff_<model_label>/`
   - `images/images_color_<model_label>/`
   - `images/images_color_perplexity_bound_<model_label>/`
   - `images/images_dots_<model_label>/`
   - `images/images_dots_perplexity_diff_<model_label>/`

2) **Run experiments (standard and Ovis)**  
   Use `run_all.ipynb` to execute both general notebooks across all tasks/models:
   - `general_notebook.ipynb`
   - `general_notebook_ovis.ipynb`

   These two notebooks exist because they rely on different imports and dependencies, so both must be executed for a complete run.

3) **Collect results**  
   All experiment outputs are saved under `data/`, preserving the per-task/per-model structure:
   - `data/data_<task>_<model_label>/`

## How the General Notebooks Work

Both `general_notebook.ipynb` and `general_notebook_ovis.ipynb` follow the same logic:

- Select the task (`asch_lines`, `color_recognition`, `dots_estimation`) and model label.
- Locate the appropriate image folders under `images/`.
- Run multiple experiment blocks (including perplexity-based conditions).
- Save intermediate and final artifacts into `data/data_<task>_<model_label>/`.

`run_all.ipynb` automates this by looping over all tasks and model labels.

## Expected Outputs

After a full run you should have:

- Images for every task/config/model in `images/`.
- Experiment results in `data/`, including per-condition pickles and summary CSVs.

## Dataset on Zenodo

The pre-generated `images/` and `data/` folders are available on Zenodo:

- DOI: `10.5281/zenodo.18022032`
- Record: `https://doi.org/10.5281/zenodo.18022032`

You can download and unpack the dataset with the script below.

```bash
bash scripts/download_zenodo.sh
```

## Minimal Quick Start (Notebook Order)

1) Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2) Run all notebooks in `create_images/` for the models you need.  
3) Run `run_all.ipynb` to generate standard + Ovis results.  
4) Inspect results in `data/`.

For exact model identifiers, see `MODEL_VERSIONS.txt`.
