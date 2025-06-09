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

## Repository structure
- **`report.pdf`** – Report which sums up whole project
- **`README.MD`** – This file
- **`project2_s340146_s336942`** – Directory stroing actual project
- **`plots`** - Directory storing generated plots used in the report
- **`tools`** - tools directory from sample_project


## Instructions

### **Example of Evaluation**
```
  python tools/evaluate_solution.py --true_file sample_test_with_ratings.csv --pred_file project1_s340146_s336942/results/pred_name.csv     
```
  you can use different files, to get better results.

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