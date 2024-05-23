import pickle
import glob
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_process_data(filepath: str,
                          data_pos: int,
                          label_pos: int) -> tuple[np.array, np.array]:
	"""
	Loads and processes data from a pickle file, balancing classes if needed.

	Args:
	filepath (str): Path to the pickle file containing the data.

	Returns:
	tuple: A tuple containing processed data and labels as numpy arrays.
	"""
	with open(filepath, 'rb') as file:
		data = pickle.load(file)


	# Extract features and labels for both classes.
	data_labeled_0 = [tup[data_pos] for tup in data if tup[label_pos] == 0]
	data_labeled_1 = [tup[data_pos] for tup in data if tup[label_pos] == 1]
	label0 = [tup[label_pos] for tup in data if tup[label_pos] == 0]
	label1 = [tup[label_pos] for tup in data if tup[label_pos] == 1]

	# Convert lists to numpy arrays.
	data0 = np.array(data_labeled_0)
	data1 = np.array(data_labeled_1)
	label0 = np.array(label0)
	label1 = np.array(label1)

	# Balance the dataset.
	if len(data0) > len(data1):
		indices = np.random.choice(len(data0), size=len(data1), replace=False)
		data0, label0 = data0[indices], label0[indices]

	if len(data1) > len(data0):
		indices = np.random.choice(len(data1), size=len(data0), replace=False)
		data1, label1 = data1[indices], label1[indices]

	print(f"Label 0 Sample Size: {len(label0)}")
	print(f"Label 1 Sample Size: {len(label1)}")


	# Combine the subsets into a single dataset.
	data = np.concatenate([data0, data1], axis=0)

	label0 = np.repeat(label0, 1)
	label1 = np.repeat(label1, 1)

	processed_labels = np.concatenate([label0, label1], axis=0)
	processed_labels[processed_labels == 2] = 1

	seq_len = data.shape[2]

	#processed_labels = np.repeat(processed_labels, 10)
	processed_data = data.reshape(-1, seq_len, 23)

	print("Current data shape:", processed_data.shape)
	print("Current labels shape:", processed_labels.shape)

	return processed_data, processed_labels


def concatenate_data_parts(folder_path: str,
                           data_pos: int,
                           label_pos: int) -> tuple[np.array, np.array]:
	"""
	Loads and concatenates data from all pickle files in a specified folder.

	Args:
	folder_path (str): The path to the folder containing pickle files.

	Returns:
	tuple: A tuple containing concatenated data and labels from all files.
	"""
	filepaths = glob.glob(f'{folder_path}/*.pkl')
	all_data, all_labels = [], []

	for file in filepaths:
		print(f"Loading {file}")
		data_part, label_part = load_and_process_data(file, data_pos, label_pos)
		all_data.append(data_part)
		all_labels.append(label_part)

	# Convert list of arrays to a single numpy array and concatenate.
	if all_data and all_labels:
		concatenated_data = np.concatenate(all_data, axis=0)
		concatenated_labels = np.concatenate(all_labels, axis=0)
		print("Total data shape:", concatenated_data.shape)
		print("Total labels shape:", concatenated_labels.shape)
		return concatenated_data, concatenated_labels
	else:
		return None, None  # Handle the case where no data is loaded.


def split_data(data: np.ndarray,
               labels: np.ndarray,
               data_path: str) -> tuple:
	"""
	Splits the data into training, validation, and test sets with stratification.

	Args:
	data (np.ndarray): The feature dataset to be split.
	labels (np.ndarray): The corresponding labels for the dataset.

	Returns:
	tuple: Contains six elements:
		- X_train (np.ndarray): Features for the training set.
		- y_train (np.ndarray): Labels for the training set.
		- X_val (np.ndarray): Features for the validation set.
		- y_val (np.ndarray): Labels for the validation set.
		- X_test (np.ndarray): Features for the test set.
		- y_test (np.ndarray): Labels for the test set.
	"""
	# First split: Separate data into training+validation set and test set.
	x_train_val, x_test, y_train_val, y_test = train_test_split(
		data, labels, test_size=0.15, random_state=31, stratify=labels
	)

	# Second split: Further split the training+validation set into training set and validation set.
	x_train, x_val, y_train, y_val = train_test_split(
		x_train_val, y_train_val, test_size=0.176, random_state=31, stratify=y_train_val
	)

	with open(f'{data_path}/test_sets/test_set_data.h5', 'wb') as f_data:
		pickle.dump(x_test, f_data)
	with open(f'{data_path}/test_sets/test_set_labels.h5', 'wb') as f_labels:
		pickle.dump(y_test, f_labels)

	return x_train, y_train, x_val, y_val, x_test, y_test


def split_data_no_test(data: np.ndarray,
               labels: np.ndarray) -> tuple:
	"""
	Splits the data into training, validation, and test sets with stratification.

	Args:
	data (np.ndarray): The feature dataset to be split.
	labels (np.ndarray): The corresponding labels for the dataset.

	Returns:
	tuple: Contains six elements:
		- X_train (np.ndarray): Features for the training set.
		- y_train (np.ndarray): Labels for the training set.
		- X_val (np.ndarray): Features for the validation set.
		- y_val (np.ndarray): Labels for the validation set.
		- X_test (np.ndarray): Features for the test set.
		- y_test (np.ndarray): Labels for the test set.
	"""

	# Second split: Further split the training+validation set into training set and validation set.
	x_train, x_val, y_train, y_val = train_test_split(
		data, labels, test_size=0.15, random_state=31, stratify=labels
	)

	return x_train, x_val, y_train, y_val