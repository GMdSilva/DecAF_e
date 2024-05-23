import random
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model

from decaf_e.attentionLayer import AttentionLayer
from decaf_e.process_data import concatenate_data_parts
from decaf_e.utils import amino_acid_key


#matplotlib.use('TkAgg')
random.seed(42)


def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def save_attention_details(save_path,
                           save_label,
                           amino_acids,
                           indices,
                           values):
    if isinstance(save_label, int):
        save_label = abs(save_label)

    print(f'Saving CSV data of {save_label} to: {save_path}/csv/{save_label}.csv')

    with open(f'{save_path}/csv/{save_label}.csv', 'w') as file:
        file.write("Amino Acid,Position,Value\n")
        for aa, pos, val in zip(amino_acids,
                                indices,
                                values):
            file.write(f"{aa},{pos},{val}\n")


def process_attention_weights(save_path,
                              positive_median_attention_weights,
                              rank,
                              type_of_test):

    n_largest_indices_per_position = np.argsort(positive_median_attention_weights, axis=1)[:, rank]

    n_largest_weights = np.array(
        [positive_median_attention_weights[i, idx] for i, idx in enumerate(n_largest_indices_per_position)])

    decoded_n_most_important_amino_acids = [amino_acid_key[index] for index in n_largest_indices_per_position]

    sequence = ''.join(decoded_n_most_important_amino_acids)

    print(f'Saving sequence of of Rank {abs(rank)} max attention weights for label {type_of_test} to: {save_path}/sequences/label_{type_of_test}_rank_{abs(rank)}_max_attention_weights_seq.txt')

    # Save the sequence to a file
    with open(f'{save_path}/sequences/label_{type_of_test}_rank_{abs(rank)}_max_attention_weights_seq.txt', 'w') as file:
        file.write(sequence)

    save_label = f'label_{type_of_test}_rank_{abs(rank)}_top_attention_weights'

    save_attention_details(save_path,
                           save_label,
                           decoded_n_most_important_amino_acids,
                           n_largest_indices_per_position,
                           n_largest_weights)


def visualize_class_attention(model_path: str,
                              results_path: str,
                              data_path: str,
                              data_pos: int,
                              label_pos: int,
                              class_target: int,
                              trial_type: str):

    # Load the model with custom attention layer
    model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})

    # Load and preprocess data
    data, labels = concatenate_data_parts(data_path, data_pos, label_pos)

    # Configure the model to retrieve predictions and attention outputs
    attention_outputs = model.get_layer('attention_layer').output
    prediction_outputs = model.get_layer('dense_1').output
    attention_model = Model(inputs=model.input, outputs=[prediction_outputs, attention_outputs])

    # Predict using the full dataset
    predictions, attention_weights = attention_model.predict(data)
    class_predictions = (predictions > 0.5).astype(int)

    # Filter data for the targeted class
    target_class_attention = attention_weights[class_predictions.flatten() == class_target]

    # Compute median attention weights and apply threshold
    median_attention = np.median(target_class_attention, axis=0)
    positive_attention = np.maximum(median_attention, 0)
    summed_attention = np.sum(positive_attention, axis=1)

    # Retrieve amino acid labels sorted by their keys
    amino_acids = sorted(amino_acid_key, key=amino_acid_key.get)

    # Plot heatmap of attention weights
    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(positive_attention.T, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    plt.yticks(range(len(amino_acids)), amino_acids)
    plt.ylabel('Amino Acid')
    plt.xlabel('Sequence Index')
    print(f'Saving Heatmap of Class {class_target} to: {results_path}/plots/{trial_type}_class_{class_target}_median_positive_attention_heatmap.png')
    plt.title(f'Median Positive Attention Heatmap for {trial_type} - Class {class_target}')
    plt.savefig(f"{results_path}/plots/{trial_type}_class_{class_target}_median_positive_attention_heatmap.png")
    plt.show()

    # Plot sum of attention weights
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(summed_attention)), summed_attention)
    plt.title(f'Summed Median Positive Attention Weights for {trial_type} - Class {class_target}')
    plt.xlabel('Sequence Index')
    plt.ylabel('Median Positive Attention Weight Sum')
    print(f'Saving Bar Plot of Class {class_target} to: {results_path}/plots/{trial_type}_class_{class_target}_summed_median_positive_attention.png')
    plt.savefig(f"{results_path}/plots/{trial_type}_class_{class_target}_summed_median_positive_attention.png")
    plt.show()

    save_label = f'label_{class_target}_summed_median_positive_attention_weights'

    rows, cols = range(len(summed_attention)), range(len(summed_attention)),
    values = summed_attention
    amino_acids = range(len(summed_attention))

    save_attention_details(results_path,
                           save_label,
                           amino_acids,
                           rows,
                           values)

    # Prepare results dataset for further processing or storage
    result_data = {
        'per_pos_attention_weights': summed_attention,
        'per_pos_per_res_transposed_attention_weights': positive_attention.T,
        'per_pos_per_res_attention_weights': positive_attention
    }

    return result_data


def compare_class_attention(model_path: str,
                          results_path: str,
                          data_path: str,
                          data_pos: int,
                          label_pos: int,
                          trial_type: str) -> None:


    results = []
    for i in (0,1):
        results.append(visualize_class_attention(model_path, results_path, data_path, data_pos, label_pos, i, trial_type))

    per_pos_0_norm = results[0]['per_pos_attention_weights'] / np.sum(results[0]['per_pos_attention_weights'])
    per_pos_1_norm = results[1]['per_pos_attention_weights'] / np.sum(results[1]['per_pos_attention_weights'])

    diff_per_pos = per_pos_0_norm - per_pos_1_norm

    plt.figure(figsize=(15, 5))
    plt.bar(range(len(diff_per_pos)), diff_per_pos)
    plt.title(f'Difference of Summed Median Attention Weights (Class 0 - Class 1)')
    plt.xlabel('Sequence Index')
    plt.ylabel('Difference')
    print(f'Saving Bar Plot of Class 0 - Class 1 diff to: {results_path}/plots/{results_path}/plots/{trial_type}_class_diff_summed_median_attention.png')
    plt.savefig(f"{results_path}/plots/{trial_type}_class_diff_summed_median_attention.png")
    plt.show()

    # Normalize attention weights across the first axis (rows) for each column
    per_pos_per_res_0_norm = results[0]['per_pos_per_res_transposed_attention_weights'] / np.sum(results[0]['per_pos_per_res_transposed_attention_weights'], axis=0)
    per_pos_per_res_1_norm = results[1]['per_pos_per_res_transposed_attention_weights'] / np.sum(results[1]['per_pos_per_res_transposed_attention_weights'], axis=0)

    # Calculate the difference in normalized weights between the two classes
    diff_per_pos_per_res = per_pos_per_res_0_norm - per_pos_per_res_1_norm
    labels = [k for k, v in sorted(amino_acid_key.items(), key=lambda item: item[1])]
    # Setup plot for heatmap of filtered weights.
    fig, ax = plt.subplots(figsize=(10, 8))

    cax = ax.imshow(diff_per_pos_per_res,
                    cmap='viridis',
                    aspect='auto',
                    interpolation='nearest')

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, rotation=0)
    ax.set_ylabel('Amino Acid')
    ax.set_xlabel('Sequence Index')
    plt.savefig(f"{results_path}/plots/{trial_type}_class_diff_2d_median_attention.png")
    print(f'Saving Heatmap of Class 0 - Class 1 diff to: {results_path}/plots/{trial_type}_class_diff_2d_median_attention.png')
    plt.show()

    process_attention_weights(results_path, results[0]['per_pos_per_res_attention_weights'], rank=-1, type_of_test=0)
    process_attention_weights(results_path, results[0]['per_pos_per_res_attention_weights'], rank=-2, type_of_test=0)
    process_attention_weights(results_path, results[1]['per_pos_per_res_attention_weights'], rank=-1, type_of_test=1)
    process_attention_weights(results_path, results[1]['per_pos_per_res_attention_weights'], rank=-2, type_of_test=1)

    # Save significant weights and corresponding labels to a file.
    rows, cols = range(len(diff_per_pos)), range(len(diff_per_pos))
    values = diff_per_pos
    amino_acids = range(len(diff_per_pos))
    positions = cols

    save_attention_details(results_path,
                           save_label='diff_label_0_label_1',
                           amino_acids=amino_acids,
                           indices=positions,
                           values=values)