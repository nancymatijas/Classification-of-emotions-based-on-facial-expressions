from data_loader_lbp import get_data_loaders
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


train_csv = "./Training_csv.csv"
val_csv = "./PublicTest_csv.csv"
test_csv = "./PrivateTest_csv.csv"

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def main():

    train_loader, val_loader, test_loader = get_data_loaders(train_csv, val_csv, test_csv, feature_extraction='lbp')

    # Priprema podataka
    X_train = np.array([x[0].numpy().flatten() if isinstance(x[0], torch.Tensor) else x[0].flatten() for x, _ in train_loader.dataset])
    y_train = np.array([y for _, y in train_loader.dataset])
    X_val = np.array([x[0].numpy().flatten() if isinstance(x[0], torch.Tensor) else x[0].flatten() for x, _ in val_loader.dataset])
    y_val = np.array([y for _, y in val_loader.dataset])
    X_test = np.array([x[0].numpy().flatten() if isinstance(x[0], torch.Tensor) else x[0].flatten() for x, _ in test_loader.dataset])
    y_test = np.array([y for _, y in test_loader.dataset])


    # RF
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    val_accuracy = rf_model.score(X_val, y_val)
    print(f'RF Validation Accuracy: {val_accuracy}')
    rf_accuracy = rf_model.score(X_test, y_test)
    print(f'RF Accuracy: {rf_accuracy}')
    rf_predictions = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, rf_predictions)
    print(f'Točnost RF: {accuracy_rf:.4f}')
    classification_rep_rf = classification_report(y_test, rf_predictions, target_names=emotions, zero_division=1)
    print(f'Izvještaj klasifikacije RF:\n{classification_rep_rf}')
    confusion_mat_rf = confusion_matrix(y_test, rf_predictions)
    print(f'Matrica zabune RF:\n{confusion_mat_rf}')

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    val_accuracy = knn_model.score(X_val, y_val)
    print(f'KNN Validation Accuracy: {val_accuracy}')
    knn_model.fit(X_train, y_train)
    knn_accuracy = knn_model.score(X_test, y_test)
    print(f'KNN Accuracy: {knn_accuracy}')
    knn_pred = knn_model.predict(X_test)
    # Metrike
    accuracy = accuracy_score(y_test, knn_pred)
    print(f'Točnost KNN: {accuracy:.4f}')
    classification_rep = classification_report(y_test, knn_pred, target_names=emotions, zero_division=1)
    print(f'Izvještaj klasifikacije KNN:\n{classification_rep}')
    confusion_mat = confusion_matrix(y_test, knn_pred)
    print(f'Matrica zabune KNN:\n{confusion_mat}')


if __name__ == "__main__":
    main()