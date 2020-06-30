import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d
from torch.optim import Adam, SGD

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.linear_layers = Sequential(
            Linear(state_size, 64),
            ReLU(inplace=True),
            Linear(64, 64),
            ReLU(inplace=True),
            Linear(64, 128),
            ReLU(inplace=True),
            Linear(128,action_size)
        )
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        state1 = self.linear_layers(state)
        return state1

        
