# PredictONCO Predictor

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10013764.svg)](https://doi.org/10.5281/zenodo.10013764)

- Web page: https://loschmidt.chemi.muni.cz/predictonco/
- Dataset on Zenodo: https://zenodo.org/doi/10.5281/zenodo.10013763

PredictONCO is the web tool for fully automated and fast analysis of the effect of mutations on stability and function in known cancer targets applying in silico methods of molecular modelling, bioinformatics and machine learning. Learn more at the [Help page](https://loschmidt.chemi.muni.cz/predictonco/help).

This repository contains the predictor code and models.
The minimal version of Python is 3.8.

## Models

The PredictONCO predictor has two models:
- **STR** - both structural and sequential features - file [xgb_struc.json](xgb_struc.json)
- **SEQ** - sequential features only - file [xgb_seq.json](xgb_seq.json)

## Run predictor

Install Python requirements.
```sh
pip3 install -r requirements.txt
```

Run predictor. You can use an example input file.
```sh
python3 predictor.py -i example_single_seq.json
```
Input file can contain a single input data object or an array of objects. For each data object, a model is chosen based on the `structure` property. Properties are defined in the `predictor.py` file as typed dictionaries.

To run predictor for the dataset published on Zenodo, use TSV conversion.
```sh
python3 tsv_convert_input.py -i PredictONCO-features.txt -o PredictONCO-features.json
python3 predictor.py -i PredictONCO-features.json -o PredictONCO-results.json
python3 tsv_convert_output.py -i PredictONCO-results.json -o PredictONCO-results.txt

# or

cat PredictONCO-features.txt | python3 tsv_convert_input.py | python3 predictor.py | python3 tsv_convert_output.py
```

## Train predictor

Install Python requirements.
```sh
pip3 install -r requirements.txt
pip3 install -r requirements_train.txt
```

Set path to the dataset in the `train_data_file` variable in [ponco_data_setup.py](ponco_data_setup.py).

Execute predictor training.
```sh
python3 ponco_train_main.py
```
