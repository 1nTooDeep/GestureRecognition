import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from config import Config
from dataset import prepare_datasets
from model import RCNN
from util.eval import eval
from util.train import train

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    seed = 1314
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Logger setting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    time = str(datetime.time(datetime.now()))[:8]
    time = time.replace(':','_')
    file_handler = logging.FileHandler(f'../log/RCNN_{time}.log')
    console_handler = logging.StreamHandler()
    file_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.addHandler(console_handler)
    epoch = 0
    model = RCNN(14)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device : {DEVICE}")
    config = Config()
    batch_size = config.config['torch_config']['batch_size']
    num_epoch = config.config['torch_config']['num_epochs']
    checkpointpath = "/home/weii/Workspace/GestureRecognition/checkpoints/RCNN/"
    if not os.path.exists(checkpointpath):
        os.mkdir(checkpointpath)
    current_epoch = 0
    num_workers = config.config['torch_config']['num_workers']
    if os.path.exists(checkpointpath) and os.listdir(checkpointpath).__len__() > 0:
        checkpoints = os.listdir(checkpointpath)
        checkpoints.sort()
        best_checkpoint_epoch = checkpoints[-1]
        model_path = os.path.join(checkpointpath, f'{best_checkpoint_epoch}')
        model.load_state_dict(torch.load(model_path))
        print(f"Loading {model_path}.")

        log.info(f"Loading {model_path}.")

    trainSet, validationSet, testSet = prepare_datasets(config, first="frame")
    trainLoader = DataLoader(
        trainSet,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    validationLoader = DataLoader(
        validationSet,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    testLoader = DataLoader(
        testSet,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    lr = config.config['torch_config']['lr']
    weight_decay = config.config['torch_config']['weight_decay']
    log.info(
        f"Using config:\nlr:{lr},\nweight_decay:{weight_decay},\nbatch_size:{batch_size},\nnum_epoch:{num_epoch},\ncheckpointpath:{checkpointpath}")

    # start training
    train(
        model,
        current_epoch,
        num_epoch,
        lr,
        weight_decay,
        trainLoader,
        validationLoader,
        log,
        checkpointpath,
        DEVICE,
        LOG_INTERVAL=50
    )
    eval(model, validationLoader, DEVICE, log)

