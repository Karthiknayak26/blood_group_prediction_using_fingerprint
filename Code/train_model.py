import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set paths
data_dir = "dataset_blood_group"
model_save_path = "new_model_testing.pkl"

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Auto-detect number of output classes
num_classes = len(dataset.classes)
print(f"Detected classes: {dataset.classes}")

# ✅ This architecture matches the one expected by app.py
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = SimpleCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load pretrained model if available
try:
    pretrained_dict = torch.load('new_model_testing.pkl', map_location=torch.device('cpu'))
    
    # Filter out unnecessary keys that are not in the current model architecture
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # Update the model's state_dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)  # Use strict=False to ignore mismatched layers
    print("✅ Pretrained model loaded successfully!")
except Exception as e:
    print(f"Error loading pretrained model: {e}")

# Train model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Save trained model
torch.save(model.state_dict(), model_save_path)
print(f"✅ Model saved as {model_save_path}")