import matplotlib
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def get_rmsd_distributions(rmsd_data_pkl: str,
                           rmsd1_tup_pos: int,
                           rmsd2_tup_pos: int):
    """
    Loads RMSD data from a pickle file containing a list of tuples

    Parameters:
    rmsd_data_pkl (str): Path to the pickle file containing the data.
    rmsd1_tup_pos (int): Position in each tuple where rmsd_1 is
    rmsd2_tup_pos (int): Position in each tuple where rmsd_2 is

    Returns: stacked_rmsd (np.array): Numpy array containing 2D RMSD data for clustering
    None
    """
    matplotlib.use('TkAgg')

    # Extract all the dgrams where the label = 0
    rmsd_list1 = [tup[rmsd1_tup_pos] for tup in rmsd_data_pkl]
    rmsd_list2 = [tup[rmsd2_tup_pos] for tup in rmsd_data_pkl]

    stacked_rmsd = np.column_stack((rmsd_list1, rmsd_list2))

    return stacked_rmsd


def cluster_and_plot(rmsd_data_2d: np.array,
                     threshold_distance: int,
                     plot_results=True):
    """
    Clusters the given RMSD data into two clusters and plots the results,
    excluding data points that are further than 'threshold_distance' from their centroids.
    Ensures consistent labeling based on proximity to the origin for each dimension and
    annotates centroids on the plot.

    Parameters:
    rmsd_data_2d (numpy array): Input array with two columns representing RMSD data.
    threshold_distance (float): Maximum distance from centroid for data points to be included in the plot.

    Returns:
    final_labels (numpy array): Labels after filtering with the threshold.
    """
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(rmsd_data_2d)
    centroids = kmeans.cluster_centers_

    # Compute the proximity order for centroids
    proximity_to_origin = np.sum(centroids**2, axis=1)
    centroid_order = np.argsort(proximity_to_origin)

    # Reorder centroids
    sorted_centroids = centroids[centroid_order]

    # Map old labels to new labels according to sorted centroids
    remap_labels = {original: new for new, original in enumerate(centroid_order)}
    sorted_labels = np.array([remap_labels[label] for label in labels])

    # Calculate distances from each point to its cluster centroid
    distances = np.linalg.norm(rmsd_data_2d - sorted_centroids[sorted_labels], axis=1)

    # Filter points where the distance to centroid is less than the threshold
    mask = distances < threshold_distance
    final_labels = np.full(sorted_labels.shape, 2)  # Initialize with 2 (dropped points)
    final_labels[mask] = sorted_labels[mask]  # Assign retained labels

    # Plotting
    if plot_results:
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'green', 'red']

        for i, color in enumerate(colors):
            if i < 2:
                plt.scatter(rmsd_data_2d[final_labels == i, 0], rmsd_data_2d[final_labels == i, 1], s=50, c=color, label=f'Cluster {i}')
            else:
                plt.scatter(rmsd_data_2d[final_labels == i, 0], rmsd_data_2d[final_labels == i, 1], s=50, c=color, label='Filtered Out')

        # Annotate centroids
        for i, centroid in enumerate(sorted_centroids):
            plt.scatter(centroid[0], centroid[1], s=300, c='red', marker='*')
            plt.annotate(f'Centroid {i}', (centroid[0], centroid[1]), textcoords="offset points", xytext=(0, 10),
                         ha='center')

        plt.title('RMSD Distribution with K-Means Clustering')
        plt.xlabel('RMSD List 1')
        plt.ylabel('RMSD List 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    return final_labels


def label_data(predictions_filepath: str,
               rmsd1_tup_pos: int,
               rmsd2_tup_pos: int,
               threshold_1: int,
               threshold_2: int,
               trial: str):
    """
    Loads a pkl file containing a list of tuples containing RMSD data for a given AF2 prediction,
    parses 2D RMSD data, groups it into two clusters (labeled 0 and 1, removing outliers), and new file with labels

    Parameters:
    rmsd_data_pkl (str): Path to the pickle file containing the data.
    rmsd1_tup_pos (int): Position in each tuple where rmsd_1 is
    rmsd2_tup_pos (int): Position in each tuple where rmsd_2 is
    threshold_1: Maximum distance from centroid for data points to be included in the plot of the first 2D RMSD set
    threshold_2: Maximum distance from centroid for data points to be included in the plot of the first 2D RMSD set

    Returns:
    final_labels (numpy array): Labels after filtering with the threshold.
    """
    files = glob.glob(predictions_filepath)

    count_label_0 = 0
    count_label_1 = 0

    for i, filename in enumerate(files):
        updated_results = []
        print(f'Parsing {filename}...')

        with open(filename, 'rb') as file:
            results = pickle.load(file)

        logits = [tup[2] for tup in results]

        rmsd_dist_1 = get_rmsd_distributions(results, rmsd1_tup_pos, rmsd1_tup_pos+2)
        rmsd_dist_2 = get_rmsd_distributions(results, rmsd2_tup_pos, rmsd2_tup_pos+2)

        labels_1 = cluster_and_plot(rmsd_dist_1, threshold_1)
        labels_2 = cluster_and_plot(rmsd_dist_2, threshold_2)


        for logit, label1, label2 in zip(logits, labels_1, labels_2):
            if label1 == 0 and label2 == 0:
                updated_results.append((logit[:1, :, :], 1))
                count_label_1 += 1
            elif label1 == 1 and label2 == 1:
                updated_results.append((logit[:1, :, :], 0))
                count_label_0 += 1

        with open(f'../af2_datasets/{trial}/part_{i}_{trial}_labeled.pkl', 'wb') as f:
            pickle.dump(updated_results, f)
    print(f'Count label 0: {count_label_0}')
    print(f'Count label 1: {count_label_1}')
