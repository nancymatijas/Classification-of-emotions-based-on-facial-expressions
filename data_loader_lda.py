import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



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

        # Define LDA instance
        self.lda = LDA()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # Convert to grayscale

        if self.feature_extraction == 'lda':
            # Apply LDA feature extraction
            feature = self.lda_fit_transform(np.array(image).reshape(-1, 48 * 48), self.data.iloc[idx, 1])
        else:
            # If no feature extraction is specified, use the image as the feature
            feature = np.array(image)

        if self.transform and self.feature_extraction != 'lda':
            # If not in the process of LDA feature extraction, apply transformations
            feature = Image.fromarray(feature)  # Convert numpy.ndarray to PIL Image
            feature = self.transform(feature)

        label = emotions.index(self.data.iloc[idx, 1])

        if self.transform and self.feature_extraction != 'lda':
            # If not in the process of LDA feature extraction, apply horizontal flip
            feature = Image.fromarray(feature)  # Convert numpy.ndarray to PIL Image
            feature = self.transform(feature)

        return feature, label

    def lda_fit_transform(self, X, y):
        # Check if there are at least two samples in the dataset
        if len(X) < 2:
            # If not, return the original data without transformation
            return X

        return self.lda.fit_transform(X, y)

    


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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return create_data_loaders(df_train, df_val, df_test, transform, feature_extraction=feature_extraction)
