# MoCaDR_project2

##  Motif finding in DNA sequences

This repository contains a simple script used to motif finding in DNA sequences. Based on user's input, script looks for motifs using EM algorithm. The report is a summary of the script's operation for various specified parameters:

- **Motif probability**

- **Motif model distribution**

- **Background model distribution**

- **Number of samples**

- **Length of DNA sequence**

Based on these parameters, script generates samples and reproduces initial distributions using the EM algorithm.
---


### Requirements  
The following libraries are required to run the code:  
- `numpy`
- `json`
- `argparse` 

## Repository structure
- **`report.pdf`** – Report which sums up whole project
- **`README.MD`** – This file
- **`project2_s340146_s336942`** – Directory storing actual project
- **`plots`** - Directory storing generated plots used in the report
- **`data`** - Directory storing data used for plots from the report

## Instructions
There are two main files: `motif_340146_336942_generate.py` and `motif_340146_336942_estimate.py`. The first one should be used for data generation and the second should be used for parameters estimation. 
### **Data generation**
Input:
- .json file with following format:
{"w": int, "alpha": 0 < float < 1, "k": int, "Theta": array of shape (4, w) of 0 < floats < 1 with columns sums up to 1, "ThetaB": list of four 0 < floats < 1 sums up to 1}

Use 
```
   python3 motif_340146_336942_generate.py --params *input --output *output_file_name
```
to generate data set with parameters from input file and save it as .json file in the format as in **Estimation** subsection.

### **Estimation**
Input:
- .json file with following format:
{"alpha": int, "X": array of shape (k, w)}

Use
```
   python3 motif_340146_336942_estimate.py --input *input --output *output_file_name
```
to estimate parameter $\boldsymbol \Theta$ from which comes the data.

### All possible parser arguments
| Argument                 | Type   | Default                             | Description                                                                                   |
|--------------------------|--------|-------------------------------------|-----------------------------------------------------------------------------------------------|
| `--train`                | str    | `"no"`                              | Whether to run training. Use `"yes"` to train, `"no"` to skip.                                |
| `--predict`              | str    | `"no"`                              | Whether to run prediction. Use `"yes"` to predict, `"no"` to skip.                            |
| `--train_file`           | str    | `"data/ratings.csv"`                | Path to your training CSV (`userId,movieId,rating`).                                          |
| `--input_file`           | str    | `"data/preds.csv"`                  | Path to your test CSV for prediction (`userId,movieId`).                                      |
| `--model_path`           | str    | `"models_trained/ALL_models.pkl"`   | Where to save (in train mode) or load (in predict mode) your pickled model data.              |
| `--output_file`          | str    | `"predictions/preds.csv"`           | Where to write your predicted ratings (`userId,movieId,rating`).                              |
| `--alg`                  | str    | `"ALL"`                             | Which algorithm to use: one of `NMF`, `SVD1`, `SVD2`, `SGD` or `ALL`.                         |
|--------------------------|--------|-------------------------------------|-----------------------------------------------------------------------------------------------|
| `--r`                    | int    | 0                                   | r to use for training (0 -> searching for best r)                                             |
| `--print_rmse_plots`     | str    | "no"                                | RMSE plot generation: 'yes' while training to save used algorithm RMSE plot                   |
| `--print_impute_plots`   | str    | "no"                                | Imputation method comparison plot generation: 'yes' while training to save used algorithm plot|
