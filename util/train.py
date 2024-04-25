import torch
import torch.nn as nn
import logging
from tabulate import tabulate
import time
from .eval import eval
from torch.cuda.amp import autocast,GradScaler

def train(model,current_epoch, num_epochs, lr, weight_decay, trainLoader, validationLoader,logger, checkpointpath, DEVICE,
          LOG_INTERVAL=200 ):
    """
    训练模型
    参数:
    - model: 要训练的模型
    - lr: 学习率
    - trainLoader: 训练数据加载器
    - logger: 日志记录器
    - DEVICE: 训练设备\
    - LOG_INTERVAL: 日志记录间隔，默认为200批次
    """
    # 设备配置
    model.to(DEVICE)
    # 优化器和损失函数配置
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    # 模型进入训练模式
    model.train()
    # 初始化统计变量
    
    for epoch in range(current_epoch,num_epochs):
        # 训练过程
        logger.info(f'In epoch: {epoch}')
        total_loss = 0.0
        correct_predictions = 0
        num_samples = 0
        batch_times = []
        for i, (inputs, labels) in enumerate(trainLoader):
            # 记录批次开始时间
            batch_start_time = time.time()

            # 数据转移到指定设备
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            output = model(inputs.to(torch.float32))
            # 计算损失
            loss = loss_fn(output, labels)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            # 更新统计变量
            loss_value = loss.item()
            total_loss += loss_value * inputs.size(0)
            _, predicted = torch.max(output, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            num_samples += inputs.size(0)

            # 记录批次结束时间并计算耗时
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)

            # 日志记录
            if i % LOG_INTERVAL == 0:
                avg_loss = total_loss / num_samples
                avg_acc = correct_predictions / num_samples * 100
                avg_batch_time = sum(batch_times) / len(batch_times)
                table_data = [
                    ["epoch","Batch", "Loss", "Accuracy", "Avg Batch Time (s)"],
                    [epoch, i, avg_loss, avg_acc, avg_batch_time]
                ]
                table_str = tabulate(table_data, headers="firstrow", floatfmt=".4f")
                logger.info(f"\n{predicted}\n{labels}")
                logger.info("\n" + table_str)
                batch_times.clear()  # 清空已记录的批次耗时列表，准备记录下一组批次耗时
        torch.save(model.state_dict(),
               checkpointpath + f'checkpoint_{epoch}.pth')
        logger.info(f"Save model in:{checkpointpath + f'checkpoint_{epoch}.pth'}")
        eval(model,validationLoader,DEVICE,logger)
