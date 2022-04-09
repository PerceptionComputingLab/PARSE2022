Baseline of PARSE 2022 challenge
==============================

The repository gives an example of how to process the PARSE challenge data (implementation based on 3D-Unet). Any other preprocessing is welcomed and any framework can be used for the challenge, the only requirement is to submit `nii.gz` files with result shape/origin/spacing/direction consistent with the original CT images (please refer to `numpy2niigz() in [`submit.py`](submit.py)`). This  repository also contains the code used to prepare the data of the challenge (data preprocessing, model training and submission result creation).

Requirements
------------
Python 3.6, PyTorch 1.6 and other common packages are listed in [`requirements.txt`](requirements.txt) or [`requirements.yaml`](requirements.yaml).

Organization
------------
Folders that aren't in the repository can be created by running the corresponding function.

    │ (3dunet)
    ├── README.md
    ├── params.pkl                            <- Example parameters for 3D-Unet
    ├── requirements.txt                      <- Txt requirements file of data processing
    ├── requirements.yaml                     <- Yaml requirements file of data processing
    ├── dataset
    │   ├── train                             <- Preprocessed training dataset folder
    │   │   ├── PA000005                      <- training data (numpy array)
    │   │  ...  ├── dcm.npy
    │   │       └── label.npy
    │   ├── eval                              <- Preprocessed testing dataset folder
    │   │   ├── PA000013/dcm.npy              <- testing data (numpy array)
    │   │  ... 
    │   ├── dcm_volume_array.npy              <- Image array of training dataset
    │   └── label_volume_array.npy            <- label array of training dataset
    ├── submit
    │   ├── npy                               <- Prediction result folder (.npy)
    │   │   ├── PA000013.npy
    │   │  ...
    │   └── nii                               <- Submission result folder (.nii.gz)
    │       ├── PA000013.nii.gz
    │      ...
    ├── exp
    │   ├── XXYY_XXYY                         <- Training exp folder
    │  ...
    ├── feature.py                            <- Available functions
    ├── dataset.py                            <- Data preprocessing and loading
    ├── model.py                              <- 3D-Unet network model
    ├── losses.py                             <- Loss functions
    ├── train.py                              <- Model training
    ├── evalu.py                              <- Model validating and testing
    ├── submit.py                             <- Submit file creation
    └── config.py                             <- Setting of file paths and training parameters

Usage
------------
The path of raw data (`root_raw_train_data` and `root_raw_eval_data`) needs to be set in [`config.py`](config.py), where raw data files need to be organized in the following way:

    │ (parse2022)
    ├── train                                 <- root_raw_train_data is set to the path
    │   ├── PA000005
    │  ...  ├── image/PA000005.nii.gz
    │       └── label/PA000005.nii.gz
    └── validate                              <- root_raw_eval_data is set to the path
        ├── PA000013/image/PA000013.nii.gz
       ...


`dataset_preprocessing()` in [`dataset.py`](dataset.py) reads and preprocesses raw data for model training and testing.

`training()` in [`train.py`](train.py) trains the model and saves the parameters, and training-related hyperparameters can be set in [`config.py`](config.py).

After `root_model_param` in [`config.py`](config.py) setting, `submit_pred()` in [`submit.py`](submit.py) can read testing data and creat submission results.

Acknowledgements
------------
- Pytorch-3dunet: https://github.com/wolny/pytorch-3dunet
- PARSE: https://github.com/XinghuaMa/PARSE
