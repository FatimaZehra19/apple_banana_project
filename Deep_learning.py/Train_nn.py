import torch 
import torch.nn as nn
import torch.optim as optim
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

#define Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*128, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, len(data_set.classes))

    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
model = SimpleNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    runing_loss = 0.0
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

        runing_loss += loss.item()
        # Accuracy calculation
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = runing_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f'Epoch [{epoch+1}/{epochs}]',
          f'Loss: {epoch_loss:.4f}', 
          f'Accuracy: {epoch_acc:.2f}%'
            )
print('Finished Training')

torch.save(model.state_dict(), 'model.pth')
print('Model saved successfully')
