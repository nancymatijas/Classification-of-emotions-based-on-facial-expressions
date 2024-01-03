from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm.notebook import tqdm


def test_model(model, test_data_loader, classes):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for data_inputs, data_labels in tqdm(test_data_loader, desc="Testing"):
           #data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)

            outputs = model(data_inputs)
            _, predicted = torch.max(outputs.data, 1)

            predictions.extend(predicted.cpu().numpy())
            ground_truth.extend(data_labels.cpu().numpy())

            total += data_labels.size(0)
            correct += (predicted == data_labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy on the test set: {100 * accuracy:.2f}%")

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
            
            # Prikaz slike
            axes[i, j].imshow(np.transpose(images[index].numpy(), (1, 2, 0)))
            axes[i, j].set_title(title, fontsize=7, y=0.95)  # Postavljanje parametra y za pomicanje teksta dolje
            axes[i, j].axis('off')

    plt.show()


