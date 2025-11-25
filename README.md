# Stroke Triage Model

This repository contains the source code and models for .... It allows for the full reproduction of the paper's results and provides a framework for applying our models to new prehospital datasets.

## Getting Started

These instructions will guide you through setting up the project environment, configuring data paths, and running the analysis pipeline.

### Prerequisites

You must have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system to manage the Python environment and its dependencies.

### Installation and Setup

Follow these steps to get your development environment running:

1.  **Clone the Repository**

    Open your terminal, navigate to the directory where you want to store the project, and clone the repository:
    ```bash
    git clone https://github.com/msabanluc/stroketriage.git
    cd stroketriage
    ```

2.  **Create and Activate the Conda Environment**

    Use the `conda-lock.yml` file to create a new Conda environment that contains all necessary packages. This command will create an environment named `stroketriage-env`.

    ```bash
    conda-lock install -n stroketriage-env conda-lock.yml
    conda activate stroketriage-env
    ```
    *(Alternatively, you can use `environment.yml`: `conda env create -f environment.yml`)*

    **Important:** After activating the environment, you must set the `PYTHONHASHSEED` environment variable to ensure reproducibility.
    ```bash
    conda env config vars set PYTHONHASHSEED=0
    conda deactivate
    conda activate stroketriage-env
    ```

3.  **Install the Project in Editable Mode**

    This crucial step makes your project's `src` folder recognizable as a Python package. This allows all scripts and notebooks to use absolute imports (e.g., `from config.config import ...`) without any further setup.

    From the root directory of the project (`stroketriage`), run:
    ```bash
    pip install -e .
    ```
    *(The `-e` stands for "editable," meaning you can change the source code without needing to reinstall the package.)*

4.  **Configure Data Paths**

    Open the `src/config.yml` file in a text editor. You must modify the directory paths to point to the location of your datasets.

    Update the following keys:
    *   `split_dir`: Path to your directory containing data splits (train, validation, test sets).
    *   `data_dir`: Path to your directory containing the raw data files.
    *   `ph2`: Path to the main dataset file.
    *   `geo`: Path to the file containing census codes for addresses.
    *   `adi`: Path to the file containing Area Deprivation Index (ADI) values.
    *   `fdc_desc`, `noc`, `prior`: Paths to tables from the SQL database (referencing ZOLL Data Systems RCSQL Data Dictionary).

    **Note:** This codebase is designed for the specific datasets used in our publication. The `fdc_desc`, `noc`, and `prior` files correspond to specific SQL tables. While the code provides a framework for analysis, using it with different prehospital data may require modifications to account for schema differences. This repository is primarily intended for reproducibility of the published results.

    Example:
    ```yaml
    # Directory for data splits.
    split_dir: "C:/Users/YourName/Documents/StrokeData/DataSplits/"
    # Directory for raw data. 
    data_dir: "C:/Users/YourName/Documents/StrokeData/Raw/"
    ```

## Running the Analysis

Once the setup is complete, you can run the analysis pipelines.

To execute the full training, evaluation, and figure generation process (orchestrating tasks across the codebase), run:
```bash
python src/scripts/run_analysis.py
```

To run the sub-analysis on ED measures (NIHSS performance), run:
```bash
python src/scripts/ed_measures_analysis.py
```

Because the project is installed as a package, you can also directly run any other script or work interactively in a notebook, and all imports will resolve correctly.

## Project Structure

-   `src/`: Contains all Python source code.
    -   `config/`: Handles project configuration and path management.
    -   `data/`: Scripts for data loading and preprocessing.
    -   `models/`: Model training, tuning, and calibration logic.
    -   `evaluation/`: Scripts for evaluating model performance and generating figures/tables.
    -   `pipelines/`: Pipelines for training and preliminary analysis.
    -   `optuna/`: JSON files containing best hyperparameters found via Optuna.
    -   `scripts/`: Executable scripts for running analyses (`run_analysis.py`, `ed_measures_analysis.py`).
-   `conda-lock.yml`: A lock file for exact environment reproduction.
-   `environment.yml`: A Conda environment file.
-   `setup.py`: Makes the project installable as a Python package.
-   `outputs/`: Default directory for saving model evaluation results (e.g., performance metrics).
-   `figures/`: Default directory for saving generated plots and figures.
