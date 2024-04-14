import os
import logging
import torch

from config import Config
from model import C3D
from torchvision.models import swin_t
from dataset import prepare_no_gabor_datasets
from util.train import train
from util.eval import eval
from util.test import test
if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    # Logger setting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    file_handler = logging.FileHandler('./C3D.log')
    console_handler = logging.StreamHandler()
    file_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.addHandler(console_handler)
    epoch = 0
    model = C3D()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device : {DEVICE}")
    config = Config()
    batch_size = config.config['torch_config']['batch_size']
    num_epoch = config.config['torch_config']['num_epochs']
    checkpointpath = "/home/weii/Workspace/GestureRecognition/checkpoints/C3D/"
    current_epoch = 0
    if os.path.exists(checkpointpath) and os.listdir(checkpointpath).__len__() > 0:
        checkpoints = os.listdir(checkpointpath)
        checkpoints.sort()
        best_checkpoint_epoch = checkpoints[-1]
        model_path = os.path.join(checkpointpath,f'{best_checkpoint_epoch}')
        print(model_path)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loading {best_checkpoint_epoch}.")

        log.info(f"Loading {best_checkpoint_epoch}.")
        current_epoch = int(best_checkpoint_epoch.replace("checkpoint_", "").replace(".pth", "")) + 1

    trainSet,validationSet,testSet = prepare_no_gabor_datasets(config)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size,
                                              shuffle=True, num_workers=config.config['torch_config']['num_workers'])
    validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=batch_size,
                                                   shuffle=True, num_workers=config.config['torch_config']['num_workers'])
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=batch_size,
                                                   shuffle=True, num_workers=config.config['torch_config']['num_workers'])
    lr = config.config['torch_config']['lr']
    weight_decay = config.config['torch_config']['weight_decay']
    train(model,current_epoch, num_epoch, lr, weight_decay, trainLoader,validationLoader, log, checkpointpath,DEVICE,LOG_INTERVAL=100)

    print(f"Save model {config.config['torch_config']['checkpoint_path'] + f'checkpoint_{epoch}.pth'}")
    torch.save(model.state_dict(),
                config.config['torch_config']['checkpoint_path'] + f'checkpoint_{epoch}.pth')
    eval(model,validationLoader,DEVICE,log)

