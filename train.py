import torch
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt


def eval_model(model, data_loader, loss_module):
    model.eval()  # Set model to eval mode
    true_preds, num_preds = 0., 0.
    total_loss = 0.0  # Initialize total loss

    with torch.no_grad():  # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:
            # Determine prediction of model on dev set
            preds = model(data_inputs)

            # Calculate loss for the current batch
            loss = loss_module(preds, data_labels)
            total_loss += loss.item()

            _, pred_labels = torch.max(preds.data, dim=1)  # Binarize predictions to 0 and 1

            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum().item()
            num_preds += data_labels.shape[0]
    
    # Calculate average loss
    avg_loss = total_loss / len(data_loader)
    acc = true_preds / num_preds

    return acc, avg_loss


def train_model(model, optimizer, scheduler, train_data_loader, val_data_loader, loss_module, num_epochs, save_model_every, model_save_path):
    model.train()   # Set model to train mode

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs): # Training loop
        running_train_loss = 0.0
        correct_train_preds = 0
        total_train_preds = 0

        for data_inputs, data_labels in tqdm(train_data_loader, 'Epoch %d' % (epoch + 1)):

            # Run the model on the input data
            preds = model(data_inputs)

            # Calculate the loss
            loss = loss_module(preds, data_labels)

            # Perform backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Accumulate training loss
            running_train_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(preds.data, 1)
            total_train_preds += data_labels.size(0)
            correct_train_preds += (predicted == data_labels).sum().item()

        # Average training loss for the epoch
        average_train_loss = running_train_loss / len(train_data_loader)
        train_losses.append(average_train_loss)

        # Training accuracy for the epoch
        train_accuracy = correct_train_preds / total_train_preds
        train_accuracies.append(train_accuracy)

        # Evaluate on the validation set
        val_accuracy, val_loss = eval_model(model, val_data_loader, loss_module)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

        # Print or log training and validation metrics
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {average_train_loss:.4f}, "
              f"Train Accuracy: {100 * train_accuracy:.2f}%, "
              f"Validation Loss: {val_loss:.4f}, "
              f"Validation Accuracy: {100 * val_accuracy:.2f}%, "
              f"Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Save the model every few epochs
        if (epoch + 1) % save_model_every == 0:
            state_dict = model.state_dict()
            torch.save(state_dict, f"{model_save_path}_epoch_{epoch + 1}.pt")

    # Save the final model after training is complete
    final_model_path = f"{model_save_path}_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at: {final_model_path}")

    # Return the training history
    return train_losses, train_accuracies, val_losses, val_accuracies


def plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # Plot Accuracy
    ax1.plot(train_accuracies, label='Train Accuracy')
    ax1.plot(val_accuracies, label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.set_title('Training and Validation Accuracy')

    # Plot Loss
    ax2.plot(train_losses, label='Train Loss')
    ax2.plot(val_losses, label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.set_title('Training and Validation Loss')

    plt.show()