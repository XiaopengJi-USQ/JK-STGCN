# JK-STGCN

The JKSTGCN model is novel sleep stage classification model based on [GraphSleepNet](https://github.com/ziyujia/GraphSleepNet) and [MSTGCN](https://github.com/ziyujia/MSTGCN).

## Dataset
The ISRUC dataset can be downloaded on the official website: https://sleeptight.isr.uc.pt/?page_id=76

## How to run

### 1. Modify configuration files
Modify configuration files especially paths in each config files.
- The `original_data_path` item in `Dataset_ISRUC_S3.json` is the path of `.mat` files
- The `label_path` item in `Dataset_ISRUC_S3.json` is the path of `.txt` label files
- The `save_path` item in `Prepocess_ISRUC_S3.json` is the path to save preporcessed data.
- The `path_preprocessed_data` item in `FeatureNet_Train.json` is the path of the preprocessed data.
- The `path_feature` item in `FeatureNet_Train.json` is the path to save extracted features by the featurenet.
- The `path_output` item in `Train.json` is the path to save models and results.
- The `path_preprocessed_data` item in `Train.json` is the path of preprocessed data.
- The `path_feature` item  in `Train.json` is the path of features.


All paths above are folders except the The `path_preprocessed_data` item in `FeatureNet_Train.json` and `path_preprocessed_data` item in `Train.json`.

### 2. Preprocess
Run `ISRUC_S3_preprocess.py`

`python ISRUC_S3_preprocess.py`

This program will preprocess the ISRUC-S3 dataset and save a file named 'ISRUC_S3.npz'.

### 3. Extract Features
Run `train_FeatureNet.py`

`python train_FeatureNet.py`

This program will extract features and save features to 'path_feature'.

### 4. Train
Run `train_JKSTGCN_model.py`

`python train_JKSTGCN_model.py`

This program will train models and save to 'path_output'.

### 5. Evaluate
Run `evaluate_JK-STGCN_model.py`

`python evaluate_JK-STGCN_model.p`

This program will evaluate models and save to 'path_output'.
