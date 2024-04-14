import pandas as pd
from .dataset_gabor import GestureDataSet
from .dataset_no_gabor import GestureDataSetNoGabor
def prepare_gabor_datasets(config):
    train_data = pd.read_csv(config.config['data_config']['train_data_csv'], sep=';')
    train_data['label'] = train_data['label'].apply(lambda x: config.label_mapping[x])

    val_data = pd.read_csv(config.config['data_config']['validation_data_csv'], sep=';')
    val_data['label'] = val_data['label'].apply(lambda x: config.label_mapping[x])

    test_data = pd.read_csv(config.config['data_config']['test_data_csv'], sep=';')
    test_data['label'] = test_data['label'].apply(lambda x: config.label_mapping[x])

    return \
        (GestureDataSet(config.config['data_config']['train_data_path'],train_data),
        GestureDataSet(config.config['data_config']['validation_data_path'], val_data),
        GestureDataSet(config.config['data_config']['test_data_path'], test_data))

def prepare_no_gabor_datasets(config):
    train_data = pd.read_csv(config.config['data_config']['train_data_csv'], sep=';')
    train_data['label'] = train_data['label'].apply(lambda x: config.label_mapping[x])

    val_data = pd.read_csv(config.config['data_config']['validation_data_csv'], sep=';')
    val_data['label'] = val_data['label'].apply(lambda x: config.label_mapping[x])

    test_data = pd.read_csv(config.config['data_config']['test_data_csv'], sep=';')
    test_data['label'] = test_data['label'].apply(lambda x: config.label_mapping[x])

    return \
        (GestureDataSetNoGabor(config.config['data_config']['train_data_path'],train_data),
        GestureDataSetNoGabor(config.config['data_config']['validation_data_path'], val_data),
        GestureDataSetNoGabor(config.config['data_config']['test_data_path'], test_data))