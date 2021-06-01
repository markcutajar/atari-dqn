import logging

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, learning_rate, checkpoint_file):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.checkpoint_file = checkpoint_file

        # The input_dims[0] corresponds to the channel
        self.conv1 = nn.Conv2d(self.input_shape[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        flattened_shape = self.calculate_flattened_shape(self.input_shape)

        self.fc1 = nn.Linear(flattened_shape, 512)
        self.fc2 = nn.Linear(512, output_shape)

        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)
        self.device = self.get_device()
        self.to(self.device)

    @staticmethod
    def get_device():
        device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_name)
        logging.info(f'Using device: {device}')
        return device

    def calculate_flattened_shape(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))

    def save_checkpoint(self):
        logging.info('Saving checkpoint')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        logging.info('Loading checkpoint')
        self.load_state_dict(torch.load(self.checkpoint_file))

    def to_tensor(self, inputs):
        return torch.tensor(inputs).to(self.device)

    def forward(self, inputs):
        # Convolutions
        x = f.relu(self.conv1(inputs))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        # Flatten
        x = x.view(x.size()[0], -1)
        # Linear layers
        x = f.relu(self.fc1(x))
        return self.fc2(x)

    def backward(self, target, value):
        loss = self.loss(target, value).to(self.device)
        loss.backward()
        self.optimizer.step()
