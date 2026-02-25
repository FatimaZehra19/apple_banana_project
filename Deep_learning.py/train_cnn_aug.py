import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Image Transformations (Clean Data)
transform = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize((128,128)), 
                                transforms.RandomHorizontalFlip(p = 0.5),               # Data Augmentation
                                transforms.RandomRotation(15),                          # Data Augmentation
                                transforms.RandomAffine(degrees=0, translate=(0.3,0)),  # Data Augmentation     
                                transforms.ToTensor()
                                ])

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
    
model = SimpleCNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')    
print('Finished Training')

# Save the model
torch.save(model.state_dict(), 'cnn_model.pth') 
print('Model saved Successfully')
