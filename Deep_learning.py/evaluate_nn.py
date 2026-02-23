import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Model Definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*128, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



# Load the trained model 
model = SimpleNN().to(device) 
model.load_state_dict(torch.load('model.pth'))
model.eval() 

print('Model loaded successfully.')



# Gaussian Noise Class (Torch version)
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0., 1.)


# Evaluation Function
def evaluate_model(transform):

    dataset = torchvision.datasets.ImageFolder(
        root="data_set",
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# Clean Transform
clean_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


# Evaluate Clean
clean_accuracy = evaluate_model(clean_transform)
print(f"Accuracy on clean data: {clean_accuracy:.2f}%")


# Evaluate Multiple Noise Levels
noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]

for std in noise_levels:

    noisy_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        AddGaussianNoise(0., std)
    ])

    noisy_accuracy = evaluate_model(noisy_transform)

    print(f"Accuracy with Gaussian noise (std={std}): {noisy_accuracy:.2f}%")