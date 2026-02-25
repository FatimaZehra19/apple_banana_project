import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Image Transformations
transform = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize((128,128)),   
                                transforms.ToTensor()])

# Load Dataset
data_set = torchvision.datasets.ImageFolder(root = "data_set", transform=transform)
train_loader = DataLoader(data_set, batch_size=32, shuffle=True)
print(f'Number of samples in dataset: {len(data_set)}')
print(f'Number of classes: {len(data_set.classes)}')


# Define Convolutional Neural Network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(16,32,3,padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*32*32, 128)
        self.fc2 = nn.Linear(128, len(data_set.classes))

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))  # 128-> 64
        x = self.pool(self.relu(self.conv2(x)))  # 64 -> 32
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load Model    
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('cnn_model.pth'))
model.eval()

# Guassian Noise Transform
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0., 1.)

# Evaluation on Noisy Data
def evaluate_model(transform):
    dataset = torchvision.datasets.ImageFolder(root="data_set", transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Clean Data Evaluation
clean_transform = transforms.Compose([transforms.Grayscale(),   
                                      transforms.Resize((128,128)),
                                      transforms.ToTensor()])
print("Evaluating on clean data...")
clean_accuracy = evaluate_model(clean_transform)
print(f"Accuracy on clean data: {clean_accuracy:.2f}%")

# Noisy Data Evaluation
noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]

for std in noise_levels:

    gaussian_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        AddGaussianNoise(0., std)
    ])

    accuracy = evaluate_model(gaussian_transform)

    print(f"Accuracy with Gaussian noise (std={std}): {accuracy:.2f}%")

# Translation Noise Evaluation
translation_transform = transforms.Compose([transforms.Grayscale(),
                                            transforms.Resize((128,128)),
                                            transforms.RandomAffine(degrees=0, translate=(0.3, 0)),  # Random translation
                                            transforms.ToTensor()
                                           ])

translation_accuracy = evaluate_model(translation_transform)
print(f"Accuracy with translation noise: {translation_accuracy:.2f}%")  

print("Evaluation completed.")
