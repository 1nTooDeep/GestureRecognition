import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import prepare_no_gabor_datasets
import config
if __name__ == '__main__':
    # Replace these values with your actual dataset, batch_size, and number of channels
    config = config.Config()
    trainSet, validationSet, testSet = prepare_no_gabor_datasets(config)

    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=8, shuffle=True, num_workers=8)
    validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=8, shuffle=True, num_workers=8)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=8, shuffle=True, num_workers=8)

    batch_size = 8
    num_channels = 3
    channels = [[], [], []]
    for frames, _ in testLoader:
        # Assuming `frames` is a tensor of shape [batch_size, num_channels, frame, height, width]
        assert frames.shape[0] == batch_size and frames.shape[1] == num_channels

        # Flatten each channel within a batch and append to corresponding list
        for batch in range(batch_size):
            if batch == 100:
                break
            for channel in range(num_channels):
                flattened_channel = frames[batch, channel, ...].flatten()
                channels[channel].append(flattened_channel)

    # Concatenate the lists to form arrays for each channel and calculate mean and std
    channel_means = []
    channel_stds = []
    for channel_data in channels:
        channel_array = np.concatenate(channel_data)
        channel_mean = np.mean(channel_array)
        channel_std = np.std(channel_array)
        channel_means.append(channel_mean)
        channel_stds.append(channel_std)

    print("Channel-wise means:", channel_means)
    print("Channel-wise standard deviations:", channel_stds)
        