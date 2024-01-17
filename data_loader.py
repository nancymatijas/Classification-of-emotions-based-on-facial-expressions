import os
import random
from shutil import copy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Define the directory path
directory_path = r'C:\Users\nancy\OneDrive\Radna površina\images_fer2013_org'
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

# DEFINING THE DIRECTORIES
path ='./'
train = 'Training'
test = 'PublicTest'


## -------------POKRENUTI SAMO PRVI PUT-------------
## Convert image data into CSV format
#columns = ['id','label']
#df_train = pd.DataFrame(columns=columns)
#df_test = pd.DataFrame(columns=columns)

## Train
#if not os.path.exists(path + 'Training_csv'):
#         os.makedirs(path + 'Training_csv')
#count = 0
#for class_name in os.listdir(train):
#     class_path = os.path.join(train, class_name)
#     for image_name in os.listdir(class_path):
#         image_path = os.path.join(class_path, image_name)
#         df_train.loc[count] = [image_name] + [class_name]
#         copy(image_path, path + 'Training_csv')
#         count += 1

## Test
#if not os.path.exists(path + 'PublicTest_csv'):
#         os.makedirs(path + 'PublicTest_csv')
#count = 0
#for class_name in os.listdir(test):
#     class_path = os.path.join(test, class_name)
#     for image_name in os.listdir(class_path):
#         image_path = os.path.join(class_path, image_name)
#         df_test.loc[count] = [image_name] + [class_name]
#         copy(image_path, path + 'PublicTest_csv')
#         count += 1
     
## Shuffling the rows in the dataframes and saving them into CSV files
#df_train = df_train.sample(frac=1).reset_index(drop=True)
#df_test = df_test.sample(frac=1).reset_index(drop=True)
#df_train.to_csv(path + "Training_csv.csv", index=False)
#df_test.to_csv(path + "PublicTest_csv.csv", index=False)

## ----------------------------------------------------


# READING THE DATA
train = './Training_csv'
test = './PublicTest_csv'
train_csv = './Training_csv.csv'
test_csv = './PublicTest_csv.csv'
df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)


emotions = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Fear', 'Disgust']
class EmotionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(train, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = emotions.index(self.data.iloc[idx, 1])
        return image, label


def visualize_data_labels(df_train):
    plt.rcParams['figure.figsize'] = (15, 5)
    df_train['label'].value_counts().plot(kind='bar')
    plt.title('Facial Emotion Data Labels', fontsize=20)
    plt.show()


def split_data(df_train):
    # Split the dataset into training, validation, and test sets (80-10-10)
    train_data, temp_data = train_test_split(df_train, test_size=0.2, random_state=602)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=602)
    return train_data, val_data, test_data


def create_data_loaders(train_data, val_data, test_data, transform):
    # Create instances of the EmotionDataset class
    train_dataset = EmotionDataset(train_data, transform=transform)
    val_dataset = EmotionDataset(val_data, transform=transform)
    test_dataset = EmotionDataset(test_data, transform=transform)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Training set size: {len(train_data)} samples")
    print(f"Validation set size: {len(val_data)} samples")
    print(f"Test set size: {len(test_data)} samples")

    return train_loader, val_loader, test_loader


def get_data_loaders(train_csv, test_csv):
    # Read the data from CSV files
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    #visualize_data_labels(df_train)
    train_data, val_data, test_data = split_data(df_train)

    # Define transformations
    #transform = transforms.Compose([
    #    transforms.Resize((64, 64)),
    #    transforms.ToTensor(),
    #])

    # Define transformations (you may need to adjust based on your requirements)
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #transforms.RandomErasing(p=0.75, scale=(0.01, 0.3), ratio=(1.0, 1.0), value=0, inplace =True)      
    ])
    return create_data_loaders(train_data, val_data, test_data, transform)



def show_sample_images(data_loader, emotions):
    # Get some random training images
    dataiter = iter(data_loader)
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


