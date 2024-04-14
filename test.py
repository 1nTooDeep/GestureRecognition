import pandas as pd
import torch
from model import C3D_L
from torch.utils.data import DataLoader
from config import Config
from dataset import GestureDataSet
import os
import logging


def test(config: Config, DEVICE, epoch):
    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 创建一个logger实例
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)  # 设置日志级别，可选DEBUG/INFO/WARNING/ERROR/CRITICAL

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler('./model.test.log')  # 指定日志文件名

    file_handler.setLevel(logging.INFO)  # 同样设置日志级别
    file_handler.setLevel(logging.DEBUG)  # 同样设置日志级别
    file_handler.setFormatter(formatter)  # 设置日志格式
    test_data = pd.read_csv(config.config['data_config']['test_data_csv'], sep=';')
    test_data['label'] = test_data['label'].apply(lambda x: config.label_mapping[x])
    test_set = train_set = GestureDataSet(config.config['data_config']['test_data_path'], test_data)
    testLoader = DataLoader(
        test_set,
        batch_size=config.config['torch_config']['batch_size'],
        shuffle=config.config['torch_config']['shuffle'],
        num_workers=config.config['torch_config']['num_workers']
    )
    model = C3D_L()
    if os.path.exists(config.config['torch_config']['checkpoint_path']) and os.listdir(
            config.config['torch_config']['checkpoint_path']).__len__() > 0:
        checkpoints = os.listdir(config.config['torch_config']['checkpoint_path'])
        checkpoints.sort()
        best_checkpoint_epoch = checkpoints[-1]
        model_path = os.path.join(config.config['torch_config']['checkpoint_path'],
                                  f'{best_checkpoint_epoch}')
        print(model_path)
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        print(f"Loading {best_checkpoint_epoch}.")
        log.info(f"Loading {best_checkpoint_epoch}.")
    else:
        model = C3D_L()
    model.to(DEVICE)
    model.eval()

    # start test, print current time
    print('Start testing... ')
    log.info('Start testing... ')
    import time
    start_time = time.time()

    total = 0
    correct = 0
    with torch.no_grad():  # 关闭梯度计算以提高效率
        for i,(test_data, test_target) in enumerate(testLoader):
            test_data = torch.ByteTensor(test_data).to(DEVICE)
            test_target = test_target.to(DEVICE)
            output = model(test_data.float())
            _, predicted = torch.max(output.data, 1)
            predicted.to(DEVICE)
            total += len(test_target)
            total += test_target.size(0)
            print(output,_,predicted,test_target,(predicted == test_target),sep='\n')
            break
            correct += (predicted == test_target).sum().item()
            if i % 100 == 0:
                print(f"In {i} batch, Accuracy: {correct/total * 100:.2f}%")
            # 计算验证集精度
        accuracy = correct / total
        print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy * 100:.2f}%')
        log.info(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    config = Config()
    DEVICE = 'cpu'
    test(config, DEVICE, 0)
