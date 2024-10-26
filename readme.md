# Agenda

- Introduction
- About the project file
- Project Setup with Miniconda

### About the notebook

The notebook is divided in 3 parts:

1. **Imports:** This section will focus on importing all necessary data and modules
1. **Data Assessing:** The focus is to familiarize with the data and assess the state of data.
1. **Questions:** This is were we solve the questions.
1. **Final Report:** This is were we answer the Assignment Question.

Please note that Data Assessing is not necessarly for answering the questions but for familiarizing with the data.

It does contains answers to some questions, but the real answer are in the part 3 _Questions_.

- Introduction
- About the project file
- Project Setup with Miniconda

---

---

## Introduction

The repository counts with the following files:

1. `cleaned.csv` This is the output of cleanning the data.
1. `ml_challenge.pdf` This is the challenge itself, its demands, requirements and questions.
1. `report.ipynb` This is where the main answer are located.
1. `requirements.txt` This is the file with all dependencies for installation. (use pip install for this).
1. `transactions.csv` This is the raw data file.

## About the project file

The project file (`report.ipynb`) is a jupyter notebook which is divided into 3 parts:

1. **Imports:** This will focus on importing all necessary data and modules.
1. **Data Assessing:** The focus is to familiarize with the data and assess the state of data.
1. **Questions:** This is were we solve the questions given by the challenge.

## Project Setup with Miniconda

This README provides instructions for setting up a Python environment using Miniconda, creating a virtual environment with Python 3.11.9, and installing the required packages for this project.

### 1. Install Miniconda

#### Windows

1. Download the Miniconda installer for Windows from the [official website](https://docs.conda.io/en/latest/miniconda.html).
2. Run the installer and follow the prompts.
3. Open the Anaconda Prompt from the Start menu.

#### macOS and Linux

1. Download the Miniconda installer for your system from the [official website](https://docs.conda.io/en/latest/miniconda.html).
2. Open a terminal and navigate to the directory containing the downloaded file.
3. Run the following command, replacing `Miniconda3-latest-MacOSX-x86_64.sh` with the name of the file you downloaded:
   ```
   bash Miniconda3-latest-MacOSX-x86_64.sh
   ```
4. Follow the prompts to complete the installation.
5. Close and reopen your terminal.

### 2. Create a Conda Environment

After installing Miniconda, create a new environment with Python 3.11.9:

```
conda create -n MyWayEnviroment python=3.11.9
```

Replace `MyWayEnviroment` with your desired environment name.

### 3. Activate the Environment

Activate the newly created environment:

```
conda activate MyWayEnviroment
```

### 4. Install Requirements

Assuming you have a `readme.txt` file with the required packages, install them using pip:

```
pip install -r readme.txt
```

If you don't have a `readme.txt` file, create one with the required packages listed one per line, then run the above command.

### 5. Verify Installation

You can verify the installation by checking the Python version and installed packages:

```
python --version
pip list
```
