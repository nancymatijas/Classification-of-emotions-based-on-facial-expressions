from train import train_model, eval_model
from test import test_model
from data_loader import get_data_loaders, show_sample_images
import torch.nn as nn
import torch.optim as optim
from model import EmotionCNN
from torch.optim.lr_scheduler import ReduceLROnPlateau


train_csv = "./Training_csv.csv"
test_csv = "./PublicTest_csv.csv"
emotions = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Fear', 'Disgust']

def main():
    train_loader, val_loader, test_loader = get_data_loaders(train_csv, test_csv)
    
    model = EmotionCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    train_model(model, optimizer, scheduler, train_loader, val_loader, criterion, num_epochs=15, save_model_every=3, model_save_path=r'C:\Users\nancy\OneDrive\Radna površina\projekt\our_model.pt')

    #train_model(model, optimizer, train_loader, val_loader, criterion, num_epochs=15, save_model_every=3, model_save_path=r'C:\Users\nancy\OneDrive\Radna površina\projekt\our_model.pt')
    test_model(model, test_loader, emotions)

if __name__ == "__main__":
    main()
