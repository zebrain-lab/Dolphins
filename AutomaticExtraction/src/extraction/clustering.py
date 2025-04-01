"""
Whistle Clustering Module

This module provides functionality for clustering similar whistle contours.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from .dtw import compute_normalized_dtw_distance

def prepare_contour_for_clustering(times, freqs):
    """
    Prepare contour data for clustering.
    
    Parameters
    ----------
    times : numpy.ndarray
        Array of time points
    freqs : numpy.ndarray
        Array of frequency points
        
    Returns
    -------
    numpy.ndarray
        2D array of [time, frequency] points
    """
    return np.column_stack((times, freqs))

def cluster_whistles(whistle_contours, distance_threshold=0.3, min_samples=2):
    """
    Cluster whistle contours based on DTW distance.
    
    Parameters
    ----------
    whistle_contours : list
        List of whistle contours, each as a 2D array of [time, frequency] points
    distance_threshold : float, optional
        Maximum DTW distance for contours to be considered in the same cluster (default=0.3)
    min_samples : int, optional
        Minimum number of samples in a cluster (default=2)
        
    Returns
    -------
    tuple
        (labels, distance_matrix) where:
        - labels: Cluster labels for each contour (-1 for noise)
        - distance_matrix: DTW distance matrix between contours
    """
    n_whistles = len(whistle_contours)
    
    # Compute distance matrix
    distance_matrix = np.zeros((n_whistles, n_whistles))
    
    for i in tqdm(range(n_whistles), desc="Computing DTW distances"):
        for j in range(i+1, n_whistles):
            distance = compute_normalized_dtw_distance(whistle_contours[i], whistle_contours[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    # Cluster using DBSCAN
    clustering = DBSCAN(eps=distance_threshold, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix)
    
    return labels, distance_matrix

def extract_cluster_representatives(whistle_contours, labels):
    """
    Extract representative contours for each cluster.
    
    Parameters
    ----------
    whistle_contours : list
        List of whistle contours, each as a 2D array of [time, frequency] points
    labels : numpy.ndarray
        Cluster labels for each contour
        
    Returns
    -------
    dict
        Dictionary mapping cluster labels to representative contours
    """
    unique_labels = np.unique(labels)
    representatives = {}
    
    for label in unique_labels:
        if label == -1:  # Skip noise
            continue
        
        # Get indices of contours in this cluster
        cluster_indices = np.where(labels == label)[0]
        
        # Find the contour with minimum average distance to others in the cluster
        min_avg_distance = float('inf')
        representative_idx = -1
        
        for i in cluster_indices:
            total_distance = 0
            for j in cluster_indices:
                if i != j:
                    distance = compute_normalized_dtw_distance(
                        whistle_contours[i], whistle_contours[j]
                    )
                    total_distance += distance
            
            avg_distance = total_distance / (len(cluster_indices) - 1) if len(cluster_indices) > 1 else 0
            
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                representative_idx = i
        
        if representative_idx >= 0:
            representatives[label] = whistle_contours[representative_idx]
    
    return representatives

def visualize_clusters(whistle_contours, labels, output_path=None):
    """
    Visualize whistle contour clusters.
    
    Parameters
    ----------
    whistle_contours : list
        List of whistle contours, each as a 2D array of [time, frequency] points
    labels : numpy.ndarray
        Cluster labels for each contour
    output_path : str, optional
        Path to save the visualization
        
    Returns
    -------
    None
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    plt.figure(figsize=(12, 8))
    
    # Plot each cluster
    for label in unique_labels:
        if label == -1:
            # Black for noise points
            color = 'k'
            marker = '.'
        else:
            # Get a color from the colormap
            color = plt.cm.viridis(float(label) / n_clusters)
            marker = 'o'
        
        # Plot contours in this cluster
        for i, contour in enumerate(whistle_contours):
            if labels[i] == label:
                plt.plot(contour[:, 0]-contour[0, 0], contour[:, 1], color=color, marker=marker, 
                         markersize=3, linestyle='-', linewidth=1, alpha=0.7)
    
    plt.title(f'Whistle Contour Clusters (n_clusters={n_clusters})')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show() 