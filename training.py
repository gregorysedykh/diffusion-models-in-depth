import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.UNet import UNet
from models.DiffusionModel import DiffusionModel

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda")           # GPU
elif torch.backends.mps.is_available():     
    device = torch.device("mps")            # Metal (for M-series Macs)
else:
    device = torch.device("cpu")            # CPU

print(f"Using device: {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")



# Load the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize((0.5,), (0.5,)),
])

# dataset = datasets.CIFAR10(root="data", download=True, transform=transform)
dataset = datasets.MNIST(root="data", download=True, transform=transform)

data_loader = DataLoader(dataset, batch_size=64, shuffle=True)


# Training parameters
learning_rate = 1e-4
epochs = 40

# Timesteps tensor
timesteps = torch.Tensor(1000).to(device)

# U-Net for 32x32 images
unet = UNet(img_channels=1)
print("Unet parameters: ", sum(p.numel() for p in unet.parameters() if p.requires_grad))

# Diffusion model
ddpm = DiffusionModel(timesteps, unet, device)
optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)


# Training loop
def train(diffusion_model, data_loader, optimizer, epochs=1):
    diffusion_model.model.train()
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        count = 0

        for images, _ in data_loader:

            # x_0 ~ q(x_0)
            images = images.to(device)

            # t ~ Uniform(1, T)
            t = torch.randint(1, diffusion_model.timesteps.shape[0], (1,))

            # eps ~ N(0, I)
            eps = torch.randn_like(images)

            # Take gradient descent step
            loss = diffusion_model.loss(images, t, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1


        average_loss = total_loss / count 
        end_time = time.time()
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {average_loss:.4f}, Time: {end_time - start_time:.2f}s')

    torch.save(diffusion_model.model.state_dict(), "unet_model_state.pth")


# Train the model
train(ddpm, data_loader, optimizer, epochs=epochs)

