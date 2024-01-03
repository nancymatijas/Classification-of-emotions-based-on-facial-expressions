import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomResizedCrop, ColorJitter
from torchvision import transforms


import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1_conv = nn.Dropout2d(0.3)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.3)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.dropout2_conv = nn.Dropout2d(0.3)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout2d(0.3)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.dropout3_conv = nn.Dropout2d(0.3)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout2d(0.3)

        # Two additional convolutional layers
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.dropout4_conv = nn.Dropout2d(0.3)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout4 = nn.Dropout2d(0.3)

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.dropout5_conv = nn.Dropout2d(0.3)
        self.bn5 = nn.BatchNorm2d(1024)
        self.dropout5 = nn.Dropout2d(0.3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(1024 * 2 * 2, 512)  # Adjust the input size based on the changes
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

        x = self.pool(F.relu(self.conv4(x)))  # Additional convolutional layer
        x = self.dropout4_conv(x)
        x = self.bn4(x)
        x = self.dropout4(x)

        x = self.pool(F.relu(self.conv5(x)))  # Additional convolutional layer
        x = self.dropout5_conv(x)
        x = self.bn5(x)
        x = self.dropout5(x)
        
        x = x.view(-1, 1024 * 2 * 2)  # Adjust the size based on the changes
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
# Augmentation and transformation
#data_transform = transforms.Compose([
#    RandomResizedCrop(64),
#    RandomHorizontalFlip(),
#    RandomRotation(10),
#    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
#])

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create an instance of the CNN model
model = EmotionCNN()
