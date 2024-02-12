import os
import pandas as pd
from shutil import copy

# Define the directory path
directory_path = r'C:\Users\nancy\OneDrive\Radna površina\images_fer2013_ds'
os.chdir(directory_path)
print("Current working directory:", os.getcwd())
contents = os.listdir()
print("Contents of the directory:", contents)

## -------------POKRENUTI SAMO PRVI PUT-------------
# DEFINING THE DIRECTORIES
path ='./'
train = 'Training'
val = 'PublicTest'
test = 'PrivateTest'

## Convert image data into CSV format
columns = ['id','label']
df_train = pd.DataFrame(columns=columns)
df_val = pd.DataFrame(columns=columns)
df_test = pd.DataFrame(columns=columns)

## Train
if not os.path.exists(path + 'Training_csv'):
         os.makedirs(path + 'Training_csv')
count = 0
for class_name in os.listdir(train):
     class_path = os.path.join(train, class_name)
     for image_name in os.listdir(class_path):
         image_path = os.path.join(class_path, image_name)
         df_train.loc[count] = [image_name] + [class_name]
         copy(image_path, path + 'Training_csv')
         count += 1

## Valid
if not os.path.exists(path + 'PublicTest_csv'):
         os.makedirs(path + 'PublicTest_csv')
count = 0
for class_name in os.listdir(val):
     class_path = os.path.join(val, class_name)
     for image_name in os.listdir(class_path):
         image_path = os.path.join(class_path, image_name)
         df_val.loc[count] = [image_name] + [class_name]
         copy(image_path, path + 'PublicTest_csv')
         count += 1
         
## Test
if not os.path.exists(path + 'PrivateTest_csv'):
         os.makedirs(path + 'PrivateTest_csv')
count = 0
for class_name in os.listdir(test):
     class_path = os.path.join(test, class_name)
     for image_name in os.listdir(class_path):
         image_path = os.path.join(class_path, image_name)
         df_test.loc[count] = [image_name] + [class_name]
         copy(image_path, path + 'PrivateTest_csv')
         count += 1

## Shuffling the rows in the dataframes and saving them into CSV files
df_train = df_train.sample(frac=1).reset_index(drop=True)
df_val = df_val.sample(frac=1).reset_index(drop=True)
df_test = df_test.sample(frac=1).reset_index(drop=True)

df_train.to_csv(path + "Training_csv.csv", index=False)
df_val.to_csv(path + "PublicTest_csv.csv", index=False)
df_test.to_csv(path + "PrivateTest_csv.csv", index=False)

