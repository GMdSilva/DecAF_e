from tensorflow.keras.models import Sequential, clone_model
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from decaf_e.process_data import concatenate_data_parts
from decaf_e.model import ModelBuilder

#matplotlib.use('TkAgg')


def train_and_save_model(model: Sequential,
                         data_path: str,
                         model_save_path: str,
                         batch_size: int = 32,
                         epochs: int = 50,
                         data_pos: int = 0,
                         label_pos: int = -1) -> None:
    """
    Trains the model using data from the specified path and saves the trained model.

    Args:
    data_path (str): The file path to the dataset.
    model_save_path (str): The file path where the trained model will be saved.
    """
    # Load and split data
    np.random.seed(42)
    data, labels = concatenate_data_parts(data_path, data_pos, label_pos)

    model.fit(data, labels, batch_size=batch_size, epochs=epochs)

    # Save the trained model
    print(f'Saving model to: {model_save_path}')
    model.save(model_save_path)


def k_fold(data_path: str,
           results_path: str,
           seq_len: str,
           dense_layers: int,
           dropout_rate: float,
           batch_size: int,
           epochs: int,
           data_pos: int = 0,
           label_pos: int = -1) -> None:
    # Load data
    X, y = concatenate_data_parts(data_path, data_pos, label_pos)

    # Initialize Stratified K-Fold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Array to store accuracy for each fold
    accuracies = []

    fold_number = 0
    # Iterate over each split
    for train_index, test_index in kf.split(X, y):  # Notice we pass both X and y
        fold_number += 1
        print(f"Training fold {fold_number}/{kf.get_n_splits()}...")

        # Split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Clone the model to ensure clean reinitialization of weights for each fold
        model = ModelBuilder(seq_len,
                         dense_layers,
                         dropout_rate).create_model()
        model.compile(optimizer=model.optimizer, loss=model.loss, metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train, y_train,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=(X_test, y_test))

        # Capture the validation accuracy
        val_accuracy = history.history['val_accuracy'][-1]
        accuracies.append(val_accuracy)

    # Print the output

        # Plot learning curves
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Fold {fold_number} - Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Fold {fold_number} - Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{results_path}/plots/model_fold_{fold_number}.png')

        plt.show()

    # Print the output
    print("Validation accuracies for each fold:", accuracies)
    print("Average validation accuracy:", np.mean(accuracies))