import os

import os
import random
import pickle
import numpy as np


def load_pkl_and_report_shape(folder_path):
    # List all .pkl files in the folder
    pkl_files = [file for file in os.listdir(folder_path) if file.endswith('.pkl')]

    # Check if there are any .pkl files
    if not pkl_files:
        print("No .pkl files found in the directory.")
        return None

    # Randomly select a .pkl file
    selected_file = random.choice(pkl_files)
    file_path = os.path.join(folder_path, selected_file)

    # Load data from the .pkl file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # Check if the loaded data is a list and contains at least one item
    if isinstance(data, list) and len(data) > 0:
        # Assume the first item in the list is a numpy array and report the shape of the first dimension
        if isinstance(data[0][0], np.ndarray):
            return data[0][0].shape[1]
        else:
            print("The first item in the list is not a numpy array.")
            return None
    else:
        print("The loaded data is not a list or is an empty list.")
        return None


def create_results_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists, not creating new folder.")


amino_acid_key = {0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I',
                  10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y',
                  19: 'V', 20: 'UNK', 21: 'GAP', 22: 'TOK'}
