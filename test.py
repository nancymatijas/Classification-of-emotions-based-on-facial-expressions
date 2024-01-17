from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error

def test_model(model, test_data_loader, classes):
    model.eval()  # Set the model to evaluation mode

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for data_inputs, data_labels in tqdm(test_data_loader, desc="Testing"):

            outputs = model(data_inputs)
            _, predicted = torch.max(outputs.data, 1)

            predictions.extend(predicted.cpu().numpy())
            ground_truth.extend(data_labels.cpu().numpy())

    # Calculate accuracy 
    accuracy_sklearn = accuracy_score(ground_truth, predictions)
    print(f"Accuracy on the test set: {100 * accuracy_sklearn:.2f}%")
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(ground_truth, predictions, average='weighted')
    recall = recall_score(ground_truth, predictions, average='weighted')
    f1 = f1_score(ground_truth, predictions, average='weighted')

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Calculate mean absolute percentage error (MAPE)
    mape = mean_absolute_error(ground_truth, predictions) / (sum(ground_truth) / len(ground_truth))
    print(f"Mean Absolute Percentage Error (MAPE): {100 * mape:.2f}%")

    # Calculate root mean square error (RMSE)
    rmse = mean_squared_error(ground_truth, predictions, squared=False)
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")

    return predictions, ground_truth


emotions = ['Angry', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Fear', 'Disgust']
def visualize_predictions(test_data_loader, classes, predictions, ground_truth, num_images=16):
    num_rows, num_cols = 4, 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    for i in range(num_rows):
        for j in range(num_cols):
            # Učitajte nove slike iz testnog skupa za svaki prikaz
            dataiter = iter(test_data_loader)
            images, labels = next(dataiter)

            # Odaberite jednu sliku iz batch-a
            index = i * num_cols + j
            title = f"GT: {emotions[labels[index]]}\nPred: {emotions[predictions[index]]}"

            # Obrnuta normalizacija
            image = images[index].numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)

            # Prikaz slike
            axes[i, j].imshow(image)
            axes[i, j].set_title(title, fontsize=7, y=0.95)
            axes[i, j].axis('off')

    plt.show()

def plot_confusion_matrix(ground_truth, predictions, emotions):
    conf_matrix = confusion_matrix(ground_truth, predictions)
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
