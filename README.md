

[![DOI](https://zenodo.org/badge/1104226102.svg)](https://doi.org/10.5281/zenodo.17727868)


# Machine Learning Models Powered by Emergency Medical Services Data Enhance Stroke Triage in Prehospital Settings

This repository contains the source code and models for our [paper](https://www.researchsquare.com/article/rs-6736104/v1) It allows for the full reproduction of the paper's results and provides a framework for applying our models to new prehospital datasets.

## Getting Started

These instructions will guide you through setting up the project environment, configuring data paths, and running the analysis pipeline.

### Prerequisites

You must have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system to manage the Python environment and its dependencies.

### Installation and Setup

Follow these steps to get the environment ready:

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/msabanluc/stroketriage-paper.git
    cd stroketriage-paper
    ```

2.  **Create and Activate the Conda Environment**

    ```bash
    conda-lock install -n stroketriage-env conda-lock.yml
    conda activate stroketriage-env
    ```
    
    *(Alternatively use `environment.yml`: `conda env create -f environment.yml`)*

    **Important:** After activating the environment, you must set the `PYTHONHASHSEED` environment variable to ensure reproducibility.
    ```bash
    conda env config vars set PYTHONHASHSEED=0
    conda deactivate
    conda activate stroketriage-env
    ```

3.  **Install the Project in Editable Mode**

    This makes the `src` folder recognizable as a Python package and allows all scripts to use absolute imports without any further setup.

    From the root directory of the project (`stroketriage`), run:
    ```bash
    pip install -e .
    ```

4.  **Configure Data Paths**

    You must modify the directory paths in `src/config.yml` to point to the location of the specific datasets.

    Update the following:
    *   `split_dir`: Path to directory containing data splits (train, validation, test sets).
    *   `data_dir`: Path to directory containing the raw data files.
    *   `ph2`: Path to the main dataset file.
    *   `geo`: Path to the file containing census codes for addresses.
    *   `adi`: Path to the file containing Area Deprivation Index (ADI) values.
    *   `fdc_desc`, `noc`, `prior`: Paths to tables from the SQL database (referencing ZOLL Data Systems RCSQL Data Dictionary).

    **Note:** This is designed for the specific datasets used in our publication. The `fdc_desc`, `noc`, and `prior` files correspond to specific SQL tables. While the code provides a framework for analysis, using it with different prehospital data may require modifications to account for differences. This repository is primarily intended for reproducibility of the published results.

## Running the Analysis

To execute the full training, evaluation, and figure generation process run:
```bash
python src/scripts/run_analysis.py
```

To run the sub-analysis on ED measures (NIHSS performance), run:
```bash
python src/scripts/ed_measures_analysis.py
```

## Project Structure

-   `src/`: Contains all source code.
    -   `config/`: Project configuration and path management.
    -   `data/`: Data loading and preprocessing.
    -   `models/`: Model training, tuning, and calibration.
    -   `evaluation/`: Evaluation of model performance and generating figures/tables.
    -   `pipelines/`: Pipelines for training and preliminary analysis.
    -   `optuna/`: JSON files containing best hyperparameters found via Optuna.
    -   `scripts/`: Executable scripts for running analyses.
    -   `outputs/`: Directory for saving model evaluation results.
    -   `figures/`: Directory for saving generated plots and figures.
-   `conda-lock.yml`: A lock file for exact environment reproduction.
-   `environment.yml`: A Conda environment file.
-   `setup.py`: Makes the project installable as a Python package.
