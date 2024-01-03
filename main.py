import torch
from train import train_model, eval_model
from test import test_model, visualize_predictions
from data_loader import get_data_loaders, show_sample_images
import torch.nn as nn
import torch.optim as optim
from model import EmotionCNN
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary


train_csv = "./Training_csv.csv"
test_csv = "./PublicTest_csv.csv"
emotions = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Fear', 'Disgust']

def main():
    train_loader, val_loader, test_loader = get_data_loaders(train_csv, test_csv)

    model = EmotionCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)
    criterion = nn.CrossEntropyLoss()    
    train_model(model, optimizer, scheduler, train_loader, val_loader, criterion, num_epochs=25, save_model_every=5, model_save_path=r'C:\Users\nancy\OneDrive\Radna površina\projekt\RUSU_ProjektPy\our_model.pt')
    predictions, ground_truth = test_model(model, test_loader, emotions)
    visualize_predictions(test_loader, emotions, predictions, ground_truth, num_images=16)


    # OVAJ DIO OTKOMENTIRATI I POZVATI AKO ZELIMO SAMO PONOVNO TESTIRATI POSTOJECI MODEL I PRIKAZAT PREDIKCIJA
    # OSTALO GORE ZAOMENTIRATI OD model = EmotionCNN()
    #model_path = r'C:\Users\nancy\OneDrive\Radna površina\projekt\our_model.pt_final.pt'
    #checkpoint = torch.load(model_path)
    #model = EmotionCNN()
    #model.load_state_dict(checkpoint)
    #model.eval()
    #predictions, ground_truth = test_model(model, test_loader, emotions)
    #visualize_predictions(test_loader, emotions, predictions, ground_truth, num_images=16)

if __name__ == "__main__":
    main()