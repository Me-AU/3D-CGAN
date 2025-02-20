# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set number of vertices per mesh and output dimension (vertices * 3 coordinates)
NUM_VERTICES = 3000
MESH_OUT_DIM = NUM_VERTICES * 3

class Generator(nn.Module):
    def __init__(self, noise_size=200, condition_size=10, hidden_dim=4096):
        super(Generator, self).__init__()
        self.noise_size = noise_size
        self.condition_size = condition_size
        self.input_size = noise_size + condition_size
        
        # Define a simple MLP network
        self.fc1 = nn.Linear(self.input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, MESH_OUT_DIM)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()  # Use tanh to output values in [-1, 1]
    
    def forward(self, noise, condition):
        # noise: (batch, noise_size), condition: (batch, condition_size)
        x = torch.cat([noise, condition], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        # Reshape to (batch, NUM_VERTICES, 3)
        x = x.view(x.size(0), NUM_VERTICES, 3)
        return x

class Discriminator(nn.Module):
    def __init__(self, condition_size=10, hidden_dim=1024):
        super(Discriminator, self).__init__()
        self.condition_size = condition_size
        
        # Input dimension: flattened mesh + condition vector (MESH_OUT_DIM + condition_size)
        self.input_dim = MESH_OUT_DIM + condition_size
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, mesh, condition):
        # mesh: (batch, NUM_VERTICES, 3) -> flatten to (batch, MESH_OUT_DIM)
        mesh_flat = mesh.view(mesh.size(0), -1)
        # Concatenate with condition: (batch, MESH_OUT_DIM + condition_size)
        x = torch.cat([mesh_flat, condition], dim=1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x.squeeze()