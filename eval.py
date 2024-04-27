import os
import logging
import torch
from config import Config
from model import Timesformer
from dataset import prepare_no_gabor_datasets
from util.timesformer_train import train
from util.timesformer_eval import eval
import numpy as np
from datetime import datetime


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    seed = 1314
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = Timesformer()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device : {DEVICE}")
    config = Config()
    batch_size = 64
    checkpointpath = "/home/weii/Workspace/GestureRecognition/checkpoints/timesformer-15/"
    num_workers = 12
    model_path = '/home/weii/Workspace/GestureRecognition/checkpoints/timesformer-15/checkpoint_49.pth'
    model.load_state_dict(torch.load(model_path))
    print(f"Loading {model_path}.")



    _,_,testSet = prepare_no_gabor_datasets(config)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)
    
    eval(model,testLoader,DEVICE,None)

