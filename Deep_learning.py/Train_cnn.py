import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device configuration
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
        x = self.pool(self.relu(self.conv1(x)))  # 128 -> 64
        x = self.pool(self.relu(self.conv2(x)))  # 64 -> 32
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


############################################
# ----------- TRAIN FUNCTION --------------
############################################

def train_model(transform, model_name):

    # Load Dataset with given transform
    data_set = torchvision.datasets.ImageFolder(
        root="data_set/Train",
        transform=transform
    )

    train_loader = DataLoader(data_set, batch_size=32, shuffle=True)

    print(f'\nTraining: {model_name}')
    print(f'Number of samples: {len(data_set)}')
    print(f'Number of classes: {len(data_set.classes)}')

    model = SimpleCNN(len(data_set.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Loss: {running_loss/len(train_loader):.4f}, '
              f'Accuracy: {100 * correct / total:.2f}%')

    print('Finished Training')

    torch.save(model.state_dict(), model_name)
    print(f'Model saved as {model_name}')


############################################
# ----------- BASELINE TRANSFORM ----------
############################################

transform_baseline = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

############################################
# ----------- NOISE TRANSFORM -------------
############################################

transform_noise = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.1)
])

############################################
# ----------- RUN EXPERIMENTS -------------
############################################

# 1️⃣ Baseline CNN
train_model(transform_baseline, "cnn_baseline.pth")

# 2️⃣ CNN with Noise Augmentation
train_model(transform_noise, "cnn_noise.pth")