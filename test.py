from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


def test_model(model, test_data_loader, criterion, classes):
    model.eval()  # Set the model to evaluation mode

    predictions = []
    ground_truth = []
    probabilities = []
    losses = []

    with torch.no_grad():
        for data_inputs, data_labels in tqdm(test_data_loader, desc="Testing"):
            outputs = model(data_inputs)
            
            # Assuming your model returns logits, calculate the loss
            loss = criterion(outputs, data_labels)
            losses.append(loss.item())

            # Assuming your model returns logits, apply softmax to get probabilities
            probabilities_batch = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            predictions.extend(predicted.cpu().numpy())
            ground_truth.extend(data_labels.cpu().numpy())
            probabilities.extend(probabilities_batch.cpu().numpy())

    # Calculate accuracy 
    accuracy_sklearn = accuracy_score(ground_truth, predictions)
    print(f"Accuracy on the test set: {100 * accuracy_sklearn:.2f}%")
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(ground_truth, predictions, average='weighted')
    recall = recall_score(ground_truth, predictions, average='weighted')
    f1 = f1_score(ground_truth, predictions, average='weighted')

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Calculate the average loss
    average_loss = sum(losses) / len(losses)
    print(f"Average Loss: {average_loss:.4f}")

    return predictions, ground_truth


emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def visualize_predictions(test_data_loader, classes, predictions, ground_truth, num_images=16):
    num_rows, num_cols = 4, 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    for i in range(num_rows):
        for j in range(num_cols):
            dataiter = iter(test_data_loader)
            images, labels = next(dataiter)
            index = i * num_cols + j
            title = f"GT: {emotions[labels[index]]}\nPred: {emotions[predictions[index]]}"

            # Obrnuta normalizacija
            image = images[index].numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)

            axes[i, j].imshow(image)
            axes[i, j].set_title(title, fontsize=7, y=0.95)
            axes[i, j].axis('off')

    plt.show()

def plot_confusion_matrix(ground_truth, predictions, emotions):
    conf_matrix = confusion_matrix(ground_truth, predictions)
    conf_matrix_percent = conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True)  
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix_percent, annot=True, fmt='.3f', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
