import logging

import torch
import torch.nn as nn

def test(model,dataLoader: torch.utils.data.DataLoader, DEVICE,log:logging.Logger):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for val_data, val_target in dataLoader:
            val_data, val_target = torch.ByteTensor(val_data).to(DEVICE), val_target.to(DEVICE)
            output = model(val_data.float())
            _, predicted = torch.max(output.data, 1)
            total += val_target.size(0)
            correct += (predicted == val_target).sum().item()

            # 计算验证集精度
        accuracy = correct / total