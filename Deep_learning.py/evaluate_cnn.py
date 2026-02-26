import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

############################################
# ----------- GAUSSIAN NOISE --------------
############################################

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


############################################
# ----------- CNN MODEL -------------------
############################################

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


############################################
# ----------- EVALUATION FUNCTION ---------
############################################

def evaluate_model(model_path, transform, description):

    print(f"\nEvaluating: {description}")

    # Load dataset
    dataset = torchvision.datasets.ImageFolder(
        root="data_set/Test",
        transform=transform
    )

    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = SimpleCNN(len(dataset.classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.2f}%")
    print("Confusion Matrix:")
    print(cm)


############################################
# ----------- CLEAN TRANSFORM -------------
############################################

transform_clean = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

############################################
# ----------- NOISY TRANSFORM -------------
############################################

transform_noisy = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.1)
])

############################################
# ----------- RUN EVALUATIONS -------------
############################################

# Evaluate Baseline on Clean
evaluate_model("cnn_baseline.pth", transform_clean, "Baseline on Clean Data")

# Evaluate Baseline on Noisy
evaluate_model("cnn_baseline.pth", transform_noisy, "Baseline on Noisy Data")

# Evaluate Noise-trained Model on Clean
evaluate_model("cnn_noise.pth", transform_clean, "Noise Model on Clean Data")

# Evaluate Noise-trained Model on Noisy
evaluate_model("cnn_noise.pth", transform_noisy, "Noise Model on Noisy Data")