import pandas as pd

from .gesture_dataset import GestureDataSet

def prepare_datasets(config,num_frames = 30,**kwargs):
    train_data = pd.read_csv(config.config['data_config']['train_data_csv'], sep=';')
    train_data['label'] = train_data['label'].apply(lambda x: config.label_mapping[x])

    val_data = pd.read_csv(config.config['data_config']['validation_data_csv'], sep=';')
    val_data['label'] = val_data['label'].apply(lambda x: config.label_mapping[x])

    test_data = pd.read_csv(config.config['data_config']['test_data_csv'], sep=';')
    test_data['label'] = test_data['label'].apply(lambda x: config.label_mapping[x])

    return \
        (GestureDataSet(config.config['data_config']['train_data_path'],train_data,num_frames = num_frames,**kwargs),
        GestureDataSet(config.config['data_config']['validation_data_path'], val_data,num_frames = num_frames,mode='validation',**kwargs),
        GestureDataSet(config.config['data_config']['test_data_path'], test_data,num_frames = num_frames,mode='validation',**kwargs))