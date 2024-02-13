import torch
from train import train_model, eval_model
from test import test_model, visualize_predictions
#from data_loader import visualize_data_labels,get_data_loaders, show_sample_images
from RUSU_ProjektPy.data_loader_hog import visualize_data_labels,get_data_loaders, show_sample_images
import torch.nn as nn
from model import EmotionCNN
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train import plot_metrics
from test import plot_confusion_matrix
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

train_csv = "./Training_csv.csv"
val_csv = "./PublicTest_csv.csv"
test_csv = "./PrivateTest_csv.csv"

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def main():
        
    df_train = pd.read_csv(train_csv)
    df_train.name = 'Train Data'
    visualize_data_labels(df_train)

    df_val = pd.read_csv(val_csv)
    df_val.name = 'Validation Data'
    visualize_data_labels(df_val)
        
    df_test = pd.read_csv(test_csv)
    df_test.name = 'Test Data'
    visualize_data_labels(df_test)


    train_loader, val_loader, test_loader = get_data_loaders(train_csv, val_csv, test_csv)
    show_sample_images(train_loader, emotions) ##vizualizacija slika iz train seta (tranformiranih)
    
    model = EmotionCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=False) 
    criterion = nn.CrossEntropyLoss()    
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(model, optimizer, 
                                                                             scheduler, train_loader, val_loader, 
                                                                             criterion, num_epochs=30, save_model_every=5, 
                                                                             model_save_path=r'C:\Users\nancy\OneDrive\Radna površina\projekt\RUSU_ProjektPy\bs64_4s',
                                                                             patience=5)
    
    predictions, ground_truth = test_model(model, test_loader, criterion, emotions)

    visualize_predictions(test_loader, emotions, predictions, ground_truth, num_images=16)
    plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses)
    plot_confusion_matrix(ground_truth, predictions, emotions)

    # OVAJ DIO OTKOMENTIRATI I POZVATI AKO ZELIMO SAMO PONOVNO TESTIRATI POSTOJECI MODEL I PRIKAZAT PREDIKCIJA
    # OSTALO GORE ZAOMENTIRATI OD model = EmotionCNN()
    #model_path = r'C:\Users\nancy\OneDrive\Radna površina\projekt\RUSU_ProjektPy\bs64_4s_final.pt'
    #checkpoint = torch.load(model_path)
    #model = EmotionCNN()
    #criterion = nn.CrossEntropyLoss()  
    #model.load_state_dict(checkpoint)
    #model.eval()
    #predictions, ground_truth = test_model(model, test_loader, criterion, emotions)
    #visualize_predictions(test_loader, emotions, predictions, ground_truth, num_images=16)
    #plot_confusion_matrix(ground_truth, predictions, emotions)
  
if __name__ == "__main__":
    main()