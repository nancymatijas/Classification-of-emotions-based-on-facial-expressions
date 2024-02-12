import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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
    def __init__(self, dataframe, directory, transform=None):
        self.data = dataframe
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = emotions.index(self.data.iloc[idx, 1])
        return image, label


def visualize_data_labels(df_train):
    plt.rcParams['figure.figsize'] = (15, 5)
    colors = ['#E64345', '#E48F1B', '#F7D027', '#6BA547', '#60CEED', '#619ED6', '#B77EA3']

    counts = df_train['label'].value_counts()
    counts.plot(kind='bar', color=colors)

    plt.title('Facial Emotion Data Labels', fontsize=15)
    plt.xlabel('Emotion', fontsize=12)

    plt.xticks(rotation=0)
    labels = counts.index
    plt.gca().set_xticklabels(labels, rotation=0, ha='center', fontsize=10)

    plt.show()


def create_data_loaders(train_data, val_data, test_data, transform):
    # Create instances of the EmotionDataset class
    train_dataset = EmotionDataset(train_data, directory=train, transform=transform)
    val_dataset = EmotionDataset(val_data, directory=val, transform=transform)
    test_dataset = EmotionDataset(test_data, directory=test, transform=transform)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Training set size: {len(train_data)} samples")
    print(f"Validation set size: {len(val_data)} samples")
    print(f"Test set size: {len(test_data)} samples")

    return train_loader, val_loader, test_loader


def get_data_loaders(train_csv, val_csv, test_csv):
    # Read the data from CSV files
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    # Define transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return create_data_loaders(df_train, df_val, df_test, transform)


def show_sample_images(train_loader, emotions):
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    images = images[:16]
    labels = labels[:16]

    num_rows, num_cols = 4, 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))

    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            title = emotions[labels[index]]
            axes[i, j].imshow(images[index].permute(1, 2, 0))
            axes[i, j].set_title(title)
            axes[i, j].axis('off')
    plt.show()

