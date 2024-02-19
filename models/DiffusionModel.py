import torch
import torch.nn as nn
from torchvision import transforms
from matplotlib import pyplot as plt
from utils import utils

import numpy as np

class DiffusionModel(nn.Module):

    def __init__(self, device, timesteps=1000):
        super().__init__()
        self.device = device
        self.timesteps = timesteps

    def forward(self, x_0):
        """
        Adds Gaussian noise to the input image using cosine scheduler
        """
        f = utils.cosine_scheduler
        s = 0.008
        alpha_prod = f(self.timesteps, self.timesteps, s) / f(0, self.timesteps, s)
        alpha_prod = torch.tensor(alpha_prod, dtype=torch.float32)

        # Check device for x_0
        if x_0.device != self.device:
            x_0 = x_0.to(self.device)

        mu = torch.sqrt(alpha_prod) * x_0
        noise = torch.sqrt(1 - alpha_prod) * torch.randn_like(x_0)

        x_t = mu + noise

        return x_t
    
    def display_forward(self, x_0):
        x_t = self.forward(x_0)
        plt.imshow(x_t)
        

    def backward(self):
        """
        Using the U-Net, the model will try to remove the noise from the image
        """
        pass
