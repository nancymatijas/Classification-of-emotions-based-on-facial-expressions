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
