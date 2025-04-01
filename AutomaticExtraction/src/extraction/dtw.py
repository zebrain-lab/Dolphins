"""
Dynamic Time Warping Module

This module provides functionality for comparing whistle contours using Dynamic Time Warping.
"""

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def compute_dtw_distance(contour1, contour2):
    """
    Compute the Dynamic Time Warping distance between two frequency contours.
    
    Parameters
    ----------
    contour1 : numpy.ndarray
        First frequency contour as a 2D array of [time, frequency] points
    contour2 : numpy.ndarray
        Second frequency contour as a 2D array of [time, frequency] points
        
    Returns
    -------
    float
        DTW distance between the contours
    """
    # Normalize time to start from 0
    c1 = np.copy(contour1)
    c2 = np.copy(contour2)
    
    c1[:, 0] -= c1[0, 0]
    c2[:, 0] -= c2[0, 0]
    
    # Use only frequency values
    c1 = c1[:, 1]
    c2 = c2[:, 1]

    # Compute DTW distance
    distance, _ = fastdtw(c1, c2)
    
    return distance

def normalize_contour(contour):
    """
    Normalize a whistle contour for comparison.
    
    Parameters
    ----------
    contour : numpy.ndarray
        Contour as a 2D array of [time, frequency] points
        
    Returns
    -------
    numpy.ndarray
        Normalized contour
    """
    # Copy the contour
    norm_contour = np.copy(contour)
    
    # Normalize time to start from 0
    norm_contour[:, 0] -= norm_contour[0, 0]
    
    # Normalize time to range [0, 1]
    if norm_contour[-1, 0] > 0:
        norm_contour[:, 0] /= norm_contour[-1, 0]
    
    # Normalize frequency to range [0, 1]
    freq_min = np.min(norm_contour[:, 1])
    freq_max = np.max(norm_contour[:, 1])
    
    if freq_max > freq_min:
        norm_contour[:, 1] = (norm_contour[:, 1] - freq_min) / (freq_max - freq_min)
    
    return norm_contour

def compute_normalized_dtw_distance(contour1, contour2):
    """
    Compute the DTW distance between two normalized frequency contours.
    
    Parameters
    ----------
    contour1 : numpy.ndarray
        First frequency contour as a 2D array of [time, frequency] points
    contour2 : numpy.ndarray
        Second frequency contour as a 2D array of [time, frequency] points
        
    Returns
    -------
    float
        Normalized DTW distance between the contours
    """
    # Normalize contours
    norm_contour1 = normalize_contour(contour1)
    norm_contour2 = normalize_contour(contour2)

    # Use only frequency values
    norm_contour1 = norm_contour1[:, 1]
    norm_contour2 = norm_contour2[:, 1]
    
    # Compute DTW distance
    distance, _ = fastdtw(norm_contour1, norm_contour2)
    
    # Normalize by contour length
    distance /= max(len(contour1), len(contour2))
    
    return distance 