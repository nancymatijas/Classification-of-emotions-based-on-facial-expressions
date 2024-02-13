import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from skimage.feature import local_binary_pattern


# Define the directory path
directory_path = r'C:\Users\nancy\OneDrive\Radna površina\images_fer2013_ds'
os.chdir(directory_path)
print("Current working directory:", os.getcwd())
contents = os.listdir()
print("Contents of the directory:", contents)

# Ensure that if you run the code multiple times, you'll get the same random outcomes
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(602)

# READING THE DATA
train = './Training_csv'
val = './PublicTest_csv'
test = './PrivateTest_csv'
train_csv = './Training_csv.csv'
val_csv = './PublicTest_csv.csv'
test_csv = './PrivateTest_csv.csv'
df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)
df_test = pd.read_csv(test_csv)

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


class EmotionDataset(Dataset):
    def __init__(self, dataframe, directory, transform=None, feature_extraction=None):
        self.data = dataframe
        self.directory = directory
        self.transform = transform
        self.feature_extraction = feature_extraction

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # Convert to grayscale

        if self.feature_extraction == 'lbp':
            # Apply LBP feature extraction
            radius = 3
            n_points = 8 * radius
            lbp_image = local_binary_pattern(np.array(image), n_points, radius, method='uniform')
            feature = lbp_image.flatten()
        else:
            # If feature extraction is not specified, use the image itself as a feature
            feature = np.array(image)

        if self.transform and self.feature_extraction != 'lbp':
            # If not in the process of extracting LBP features, apply transformations
            feature = self.transform(feature)

        label = emotions.index(self.data.iloc[idx, 1])

        if self.transform and self.feature_extraction != 'lbp':
            # If not in the process of extracting LBP features, apply horizontal flip
            feature = self.transform(feature)

        return feature, label


def create_data_loaders(train_data, val_data, test_data, transform, feature_extraction=None):
    # Create instances of the EmotionDataset class
    train_dataset = EmotionDataset(train_data, directory=train, transform=transform, feature_extraction=feature_extraction)
    val_dataset = EmotionDataset(val_data, directory=val, transform=transform, feature_extraction=feature_extraction)
    test_dataset = EmotionDataset(test_data, directory=test, transform=transform, feature_extraction=feature_extraction)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Training set size: {len(train_data)} samples")
    print(f"Validation set size: {len(val_data)} samples")
    print(f"Test set size: {len(test_data)} samples")

    return train_loader, val_loader, test_loader


def get_data_loaders(train_csv, val_csv, test_csv, feature_extraction=None):
    # Read the data from CSV files
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    # Define transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # Normalize differently for grayscale and RGB images
        transforms.Normalize([0.5], [0.5]) if feature_extraction == 'lbp' else
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return create_data_loaders(df_train, df_val, df_test, transform, feature_extraction='lbp')


