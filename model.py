import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomResizedCrop, ColorJitter
from torchvision import transforms

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1_conv = nn.Dropout2d(0.5)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.5)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.dropout2_conv = nn.Dropout2d(0.5)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout2d(0.5)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.dropout3_conv = nn.Dropout2d(0.5)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout2d(0.5)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1_conv(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2_conv(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3_conv(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Augmentation and transformation
data_transform = transforms.Compose([
    RandomResizedCrop(64),
    RandomHorizontalFlip(),
    RandomRotation(10),
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
])

# Create an instance of the CNN model
model = EmotionCNN()