# Conformity LLMs: Image-Based Task Pipeline

This repository generates synthetic images for three perceptual tasks and runs LLM-based experiments to evaluate performance and conformity patterns. The workflow is notebook-driven and split into two experiment paths (standard and Ovis) because they require different imports and dependencies.

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

## Minimal Quick Start (Notebook Order)

1) Run all notebooks in `create_images/` for the models you need.  
2) Run `run_all.ipynb` to generate standard + Ovis results.  
3) Inspect results in `data/`.
