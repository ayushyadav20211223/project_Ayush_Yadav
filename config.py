import torch.nn as nn


batchsize = 32
epochs = 5
resize_x, resize_y = 64, 64
input_channels = 3
num_classes = 2
learning_rate = 0.001
data_path = 'data/'

loss_fn = nn.CrossEntropyLoss()