# Pusakaofjava Indonesia Corp Auto-AI Program

## Overview
This repository contains an Auto-AI program developed by Pusakaofjava Indonesia Corp. The program involves creating a POJiC69 AI model, generating datasheets, combining data, and training the Auto-AI with a progress bar.

## Requirements
- Python 3.6 or later
- torch
- pandas
- tqdm
- numpy

## Installation
```bash
pip install torch pandas tqdm numpy
```

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/adearya69/PusakaOfJava.git
    cd PusakaOfJava
    ```
2. Run the program:
    ```bash
    python3 POJiC69-AI.py
    ```

## Description
The program consists of the following components:

### 1. POJiC69 AI Model
The `POJiC69Model` class defines a simple linear neural network for the Auto-AI.

### 2. Data Generation
Datasheets are generated with progress bars, and concrete data is added to 'Matrix-4thy.csv' and 'LOV-e.Alhg.csv'.

### 3. Auto-AI Training
Datasets are loaded, and the Auto-AI model is trained with a progress bar using the MSELoss criterion and Adam optimizer.

### 4. Error Handling
The program checks for file existence and empty datasets, raising appropriate errors.

## Important Notes
- Adjust the concrete data size in the code based on your requirements.
- Make sure to install the required dependencies before running the program.

## Contact
For inquiries or support, please contact Pusakaofjava Indonesia Corp at [kreatifi@kreatifindonesia.com](mailto:kreatifi@kreatifindonesia.com).
