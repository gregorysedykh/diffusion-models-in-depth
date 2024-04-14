import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel:
    def __init__(self, timesteps: torch.Tensor, model, device):
        """
        Args:
            timesteps: torch.Tensor - tensor of shape (timesteps,) containing the number of timesteps
            model: nn.Module - model to be used for the diffusion process
            device: torch.device - device to be used
        """

        self.device = device
        self.model = model.to(self.device)
        self.timesteps = timesteps

        # Forward process parameters
        self.beta_linear_schedule = torch.linspace(10**(-4), 0.02, timesteps.shape[0]).to(self.device)
        self.alpha_schedule = 1 - self.beta_linear_schedule
        self.alpha_prod = torch.cumprod(self.alpha_schedule, dim=0)
    
        self.sigma2 = self.beta_linear_schedule


    def forward_process(self, x_0, t, noise=None):
        """
        Input:
            x_0: tensor of shape (batch_size, 3, H, W) - image to be noised
            t: int - timestep
        Output:
            x_t: tensor of shape (batch_size, 3, H, W) - noised image for the given timestep t
        """
        mean = torch.sqrt(self.alpha_prod[t]) * x_0
        std = torch.sqrt(1 - self.alpha_prod[t])
        if noise is None:
            noise = torch.randn_like(x_0) * std
        x_t = mean + noise * std

        return x_t


    def loss(self, x_0, t, eps):
        """
        Input:
            x_0: tensor of shape (batch_size, 3, H, W) - image
            t: tensor of shape (batch_size,) - timestep
            eps: tensor of shape (batch_size, 3, H, W) - noise
        Output:
            returns the loss for the given timestep t
        """
        x_t = self.forward_process(x_0, t, eps)
        loss = F.mse_loss(self.model(x_t, t), eps)

        return loss
    

    def sample(self):
        """
        Samples a random image from the model.
        """
        self.model.eval()
        with torch.no_grad():
            x_t = torch.randn(64, 1, 32, 32).to(self.device)

            for t in reversed(range(self.timesteps.shape[0])):
                if t > 1:
                    z = torch.randn_like(x_t)
                else:
                    z = torch.zeros_like(x_t)

                # Compute scaling factors and noise contributions
                sqrt_alpha_t_inv = 1 / torch.sqrt(self.alpha_schedule)
                sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - self.alpha_prod)
                sqrt_beta_t = torch.sqrt(self.sigma2)

                # Predict noise using the model
                predicted_noise = self.model(x_t, torch.tensor([t-1]))

                # Denoise step
                x_t = sqrt_alpha_t_inv[t-1] * (x_t - (1 - self.alpha_schedule[t-1])/(sqrt_one_minus_alpha_prod_t[t-1]) * predicted_noise) + sqrt_beta_t[t-1] * z

            return x_t


