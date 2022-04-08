class DefaultConfig (object) :

    root_raw_train_data = 'path to be filled' # downloaded training data root
    root_raw_eval_data = 'path to be filled' # downloaded evaluation/testing data root

    root_dataset_file = './dataset/' # preprocessed dataset root
    root_train_volume = './dataset/train/' # preprocessed training dataset root
    root_eval_volume = './dataset/eval/' # preprocessed evaluation/testing dataset root

    root_exp_file = './exp/' # training exp root
    root_submit_file = './submit/' # submit file root

    root_pred_dcm = 'path to be filled' # prediction array root
    root_pred_save = 'path to be filled' # prediction result root

    root_model_param = './params.pkl' # reference model parameter root

    use_gpu = True
    batch_size = 2
    max_epoch = 1
    learning_rate = 1e-3
    decay_LR = (5, 0.1)

opt  = DefaultConfig()