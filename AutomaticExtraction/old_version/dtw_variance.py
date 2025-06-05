#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whistle Detection and Analysis Module

This module provides functions for detecting and analyzing whistles in spectrograms,
particularly useful for bioacoustic analysis. It includes tools for variance analysis,
whistle detection, and comparison with annotated whistles using Dynamic Time Warping.

"""

import numpy as np
import cv2
from fastdtw import fastdtw
from joblib import Parallel, delayed
from scipy.signal import find_peaks
from tqdm import tqdm
import pandas as pd
from .spectrogram import wav_to_spec

def merge_consecutive_windows(df):
    """
    Merge consecutive detection windows in a DataFrame.
    
    Args:
        df: pandas DataFrame with columns 'file_name', 'initial_point', 'finish_point', 'confidence'
    
    Returns:
        pandas DataFrame with merged consecutive windows
    """
    # Initialize lists to store merged results
    merged_records = []
    current_record = None
    
    for _, row in df.iterrows():
        if current_record is None:
            current_record = {
                'file_name': row['file_name'],
                'initial_point': row['initial_point'],
                'finish_point': row['finish_point'],
                'confidence': row['confidence']
            }
        else:
            # Check if this window is consecutive with the previous one
            if row['initial_point'] == current_record['finish_point']:
                # Update the finish point and average the confidence
                current_record['finish_point'] = row['finish_point']
                current_record['confidence'] = (current_record['confidence'] + row['confidence']) / 2
            else:
                # Save the current record and start a new one
                merged_records.append(current_record)
                current_record = {
                    'file_name': row['file_name'],
                    'initial_point': row['initial_point'],
                    'finish_point': row['finish_point'],
                    'confidence': row['confidence']
                }
    
    # Add the last record if it exists
    if current_record is not None:
        merged_records.append(current_record)
    
    # Create and return a new DataFrame with merged records
    return pd.DataFrame(merged_records)

def variance_vector(specgram, freqs, window_size=5, n_jobs=-1):
    """
    Compute variance of frequency intensities in sliding windows across a spectrogram.
    
    Parameters
    ----------
    specgram : numpy.ndarray
        The spectrogram matrix (frequency x time)
    freqs : numpy.ndarray
        Array of frequency values
    window_size : int, optional
        Size of the sliding window (default=5)
    n_jobs : int, optional
        Number of parallel jobs (-1 for all cores) (default=-1)
        
    Returns
    -------
    numpy.ndarray
        Vector of variance values for each window position
    """
    # Initialize the vector and the window's indexes
    var_wind = np.zeros(specgram.shape[1])
    windows = []
    index_window_min = 0
    index_window_max = window_size
    
    # Create list of all window positions
    windows.append((index_window_min, index_window_max))
    while index_window_max < specgram.shape[1]:
        index_window_min += 1
        index_window_max += 1
        windows.append((index_window_min, index_window_max))
    
    # Add padding windows at the end
    [windows.append((index_window_min, index_window_max)) for _ in range(4)]
    
    def apply_variance(window):
        """Helper function to compute variance for a single window"""
        # Sum frequency intensities in the window
        current_window = np.apply_along_axis(np.sum, 1, specgram[:,window[0]:window[1]])
        
        # Convert to probability distribution
        current_window = current_window/sum(current_window)
        
        # Calculate variance using expectation formula
        expectation = sum(current_window*freqs)
        expectation_squared = sum(current_window*(freqs**2))
        return expectation_squared - expectation**2
    
    # Compute variance for all windows in parallel
    var_wind = Parallel(n_jobs=n_jobs)(delayed(apply_variance)(window) for window in tqdm(windows))
    
    return np.array(var_wind)

def whistle_zones(var_wind, threshold=1e6, window_length=60, selection_percentage=0.20):
    """
    Identify regions of low variance that likely contain whistles.
    
    Parameters
    ----------
    var_wind : numpy.ndarray
        Vector of variance values
    threshold : float, optional
        Variance threshold for whistle detection (default=1e6)
    window_length : int, optional
        Length of analysis window (default=60)
    selection_percentage : float, optional
        Minimum percentage of low variance points needed (default=0.20)
        
    Returns
    -------
    numpy.ndarray
        Binary array marking whistle zones (1) and non-whistle zones (0)
    """
    # Create binary labels where 1 indicates variance below threshold
    labels = (var_wind <= threshold)*1
    
    # Initialize output array
    wh_zone = np.zeros(len(labels), dtype=int)
    
    # Sliding window parameters
    index_window_min = 0
    index_window_max = window_length
    
    # Loop until the window passes outside the labels vector
    while index_window_max < len(labels):
        # Mark as whistle zone if enough low variance points are present
        if sum(labels[index_window_min:index_window_max]) >= selection_percentage*window_length:
            wh_zone[index_window_min:index_window_max] = 1
        
        index_window_min += 1
        index_window_max += 1
    
    # Process final window
    if sum(labels[index_window_min:index_window_max]) >= selection_percentage*window_length:
        wh_zone[index_window_min:index_window_max] = 1
        
    return wh_zone

def spectral_subtraction(specgram, noise_estimation_factor=1.5):
    """
    Apply spectral subtraction to remove noise from the spectrogram.

    Parameters
    ----------
    specgram : numpy.ndarray
        The spectrogram matrix (frequency x time)
    noise_estimation_factor : float
        Factor to estimate noise level (default=1.5)

    Returns
    -------
    numpy.ndarray
        The spectrogram with noise removed
    """
    # Estimate noise level (you can adjust this based on your recordings)
    noise_estimate = np.mean(specgram, axis=1, keepdims=True) * noise_estimation_factor

    # Subtract the estimated noise from the spectrogram
    filtered_specgram = np.maximum(specgram - noise_estimate, 0)  # Ensure no negative values

    return filtered_specgram

def filter_edges_spectrogram(specgram, threshold1=50, threshold2=100):

    D_db = 10 * np.log10(np.abs(specgram + 1e-14))
    # Normalize spectrogram for OpenCV processing
    D_norm = cv2.normalize(D_db, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Apply Gaussian Blur to reduce noise
    D_blur = cv2.GaussianBlur(D_norm, (7, 7), 0)
    # Apply Canny Edge Detection
    edges = cv2.Canny(D_blur, threshold1=threshold1, threshold2=threshold2)
    # Create a mask from the edges
    mask = edges > 0
    # Remove values outside of the edges in the original specgram
    specgram_filtered = np.zeros_like(specgram)
    # Apply the mask to the specgram
    specgram_filtered[mask] = specgram[mask]
    return specgram_filtered

def vectorize_wh_zones(audio_path, detections_df, window_size=5, delta=3, min_freq=3000, max_freq=22000, noise_reduction=False):
    """
    Extract frequency-time points from whistle zones in an audio file.
    
    Parameters
    ----------
    audio_path : str
        Path to the audio file
    detections_df : pandas.DataFrame
        DataFrame with columns 'initial_point', 'finish_point', 'confidence'
        containing the detected whistle windows
    window_size : int, optional
        Size of sliding window in time bins (default=5)
    delta : int, optional
        Step size for sliding window (default=3)
    min_freq : int, optional
        Minimum frequency to analyze in Hz (default=3000)
    max_freq : int, optional
        Maximum frequency to analyze in Hz (default=22000)
    noise_reduction : bool, optional
        Whether to apply noise reduction to the spectrogram (default=False)
        
    Returns
    -------
    tuple
        Two arrays containing:
        - wht: Times of detected whistles
        - whf: Frequencies of detected whistles
    """
    # Initialize lists for whistle times and frequencies
    whf = []
    wht = []
    
    # Skip processing if no detections
    if len(detections_df) == 0:
        return np.array([]), np.array([])
    
    # Progress bar
    pbar = tqdm(total=len(detections_df))
    
    # Process each detection window
    for _, row in detections_df.iterrows():
        # Get spectrogram only for the detection window
        window_start = row['initial_point']
        window_end = row['finish_point']
        
        # Compute spectrogram for this segment
        specgram, freqs, times = wav_to_spec(
            audio_path,
            stride_ms=10.0,
            window_ms=20.0,
            max_freq=max_freq,
            min_freq=min_freq,
            cut=(window_start, window_end)
        )

        if noise_reduction:
            specgram = spectral_subtraction(specgram)

        specgram = filter_edges_spectrogram(specgram)

        # Lists for this detection window
        window_whf = []
        window_wht = []
        
        # Process the window in smaller chunks
        index_window_min = 0
        index_window_max = window_size
        
        while index_window_max < specgram.shape[1]:
            # Sum frequency intensities in the current window
            current_window = np.apply_along_axis(np.sum, 1, 
                                               specgram[:, index_window_min:index_window_max])
            
            # Get the time for this window
            t = times[index_window_min]
            
            if t not in wht:
                window_whf.append(freqs[np.argmax(current_window)])
                window_wht.append(t)
            
            # Move the window
            index_window_min += 1
            index_window_max += 1
        
        # Smooth the frequency contour for this window
        if len(window_whf) > 1:
            # Get time step from the data
            time_step = np.median(np.diff(window_wht))
            delta_time = 5 * time_step
            
            # Convert to numpy arrays for processing
            window_whf = np.array(window_whf)
            window_wht = np.array(window_wht)
            
            # Process each point in this window
            for i in range(len(window_whf)):
                # Get neighboring points within delta_time
                time_diffs = np.abs(window_wht - window_wht[i])
                nearby_mask = time_diffs <= delta_time
                
                if np.sum(nearby_mask) > 1:  # If there are nearby points
                    nearby_freqs = window_whf[nearby_mask]
                    # Define frequency search range
                    freq_range = np.max(nearby_freqs) - np.min(nearby_freqs)
                    search_window = (np.min(nearby_freqs) - delta * freq_range,
                                   np.max(nearby_freqs) + delta * freq_range)
                    
                    # Get the time index in this window's spectrogram
                    time_idx = np.where(times == window_wht[i])[0][0]
                    
                    # Extract the relevant portion from the spectrogram
                    freq_mask = (freqs >= max(search_window[0], min_freq)) & (freqs <= min(search_window[1], max_freq))
                    point_specgram = specgram[freq_mask, time_idx:time_idx + window_size]
                    point_freqs = freqs[freq_mask]
                    
                    if point_specgram.size > 0:
                        # Find the strongest frequency in this range
                        power_spectrum = np.sum(point_specgram, axis=1)
                        window_whf[i] = point_freqs[np.argmax(power_spectrum)]
        
        # Add the smoothed window data to the main lists
        whf.extend(window_whf)
        wht.extend(window_wht)
        
        # Update progress bar
        pbar.update(1)
    
    pbar.close()
    
    # Convert to numpy arrays
    wht = np.array(wht)
    whf = np.array(whf)
    
    # Sort by time
    sort_idx = np.argsort(wht)
    wht = wht[sort_idx]
    whf = whf[sort_idx]
    
    return wht, whf

def split_wh_zones(wht, whf, delta_t = 1):
    """
    Split whistle detections into separate segments based on time gaps.
    
    Parameters
    ----------
    wht : numpy.ndarray
        Array of whistle time points
    whf : numpy.ndarray
        Array of whistle frequencies
    delta_t : float, optional
        Minimum time gap to split whistles (default=1 second)
        
    Returns
    -------
    list
        List of tuples containing (times, frequencies) for each whistle segment
    """
    # List to store separated whistle zones
    split_wh = []
    
    # Initialization of current whistle zone
    current_whf = [whf[0]]
    current_wht = [wht[0]]
    
    # Loop until out of whole whistle zone
    i = 1
    while (i < len(wht)):
        # Group points that are closer than delta_t
        while ((wht[i]-wht[i-1] < delta_t) & (i < len(wht))) :
            current_whf.append(whf[i])
            current_wht.append(wht[i])
            if i == len(wht)-1: break
            i+=1
        
        # Add current whistle zone to list
        split_wh.append((current_wht, current_whf))
        
        # Initialize new whistle zone
        current_whf = [whf[i]]
        current_wht = [wht[i]]
        
        i+=1
    
    return split_wh

def distance_wh_zones(wht, whf, annotation_freqs, annotation_times, alphas = (0.5,1.25), n_jobs=-1):
    """
    Calculate DTW distances between detected whistles and annotation template.
    
    Parameters
    ----------
    wht : numpy.ndarray
        Whistle time points
    whf : numpy.ndarray
        Whistle frequencies
    annotation_freqs : numpy.ndarray
        Template annotation frequencies
    annotation_times : numpy.ndarray
        Template annotation times
    alphas : tuple, optional
        (min_scale, max_scale) for window size relative to annotation length (default=(0.5,1.25))
    n_jobs : int, optional
        Number of parallel jobs (default=-1)
        
    Returns
    -------
    tuple
        (raw distances array, normalized distances array)
    """
    # List to store every window
    windows = []
    
    # Get duration of annotation
    len_annots_times = annotation_times[-1] - annotation_times[0]
    
    # First window from t=0 to t= alpha1 (default:1.25) x annotation duration
    ind_min = 0
    ind_max = sum((wht - wht[0]) <= alphas[1]*len_annots_times)
    windows.append((ind_min, ind_max))
    ind_min += 1
    
    # Loop until window is out of spectrogram    
    while (ind_max < len(wht))&(ind_min < len(wht)) :
        # Select right border of window such as window length <= alpha1 x annotation duration 
        ind_max = np.max(np.where(wht <= wht[ind_min]+alphas[1]*len_annots_times))
        
        # Test if window length >= alpha0 x annotation duration
        if wht[ind_max] - wht[ind_min] >= alphas[0]*len_annots_times:
            windows.append((ind_min, ind_max))
            ind_min += 1
        else:
            # If window length < alpha0 x annotation duration
            n_points = ind_max - ind_min
            [windows.append((ind_min, ind_max)) for _ in range(n_points+1)]
            ind_min = ind_max+1
    
    def apply_distance(window):
        """Helper function to compute DTW distance for a window"""
        dist, _ = fastdtw(annotation_freqs, whf[window[0]:window[1]])
        norm_dist = dist/(window[1]-window[0])
        return dist, norm_dist
    
    # Compute distances in parallel
    Distances = Parallel(n_jobs=n_jobs)(delayed(apply_distance)(window) for window in tqdm(windows))
    
    Dist = [d[0] for d in Distances]
    Norm_Dist = [d[1] for d in Distances]
    
    return np.array(Dist), np.array(Norm_Dist)

def distance_per_frame(recording, times, annotation_freqs, annotation_times, n_jobs=-1):
    """
    Compute DTW distance between each frame of the recording and the annotation template.
    
    Parameters
    ----------
    recording : numpy.ndarray
        Frequency vector of the recording
    times : numpy.ndarray
        Time vector of the recording
    annotation_freqs : numpy.ndarray
        Template annotation frequencies
    annotation_times : numpy.ndarray
        Template annotation times
    n_jobs : int, optional
        Number of parallel jobs (default=-1)
        
    Returns
    -------
    numpy.ndarray
        Array of DTW distances for each frame
    """
    # Time to start at t=0
    var_annots_times = annotation_times - annotation_times[0]
    var_times = times - times[0]
    
    # List to store each frame
    windows = []
    
    # First frame
    ind_min = 0
    ind_max = sum(var_times <= var_annots_times[-1])
    windows.append((ind_min, ind_max))
    
    ind_min +=1
    ind_max +=1
    
    # Create sliding windows
    while (ind_min < len(recording)):
        windows.append((ind_min, ind_max))
        ind_min +=1
        ind_max +=1
    
    def apply_distance(window):
        """Helper function to compute DTW distance for a window"""
        dist, _ = fastdtw(annotation_freqs, recording[window[0]:window[1]])
        return dist
    
    # Compute distances in parallel
    Dist = Parallel(n_jobs=n_jobs)(delayed(apply_distance)(window) for window in tqdm(windows))
    
    return np.array(Dist)

def get_whistle_limit(annotation_times, annotation_freqs, wh_times, wh_freqs):
    """
    Detect whistle's start and end points using DTW points association.

    Parameters
    ----------
    annotation_times : numpy.ndarray
        Time points of the annotation template
    annotation_freqs : numpy.ndarray
        Frequency values of the annotation template
    wh_times : numpy.ndarray
        Time points of the detected whistle
    wh_freqs : numpy.ndarray
        Frequency values of the detected whistle

    Returns
    -------
    tuple
        (trimmed whistle frequencies, trimmed whistle times)
    """
    # Calculate percentile points for analysis (10% and 90%)
    first = int(np.ceil(10/100*len(annotation_freqs)))
    last = int(np.floor(90/100*len(annotation_freqs)))
    
    # Compute DTW path between annotation and whistle
    dist, path = fastdtw(annotation_freqs, wh_freqs)
    
    # Extract path coordinates
    p0 = np.array([t[0] for t in path])
    p1 = np.array([t[1] for t in path])
    
    # Map annotation points to whistle points
    s = [0]
    for i in range(len(annotation_freqs)):
        s.append(p1[max(np.where(p0==i)[0])])
    
    s = np.array(s)
    
    # Calculate point-to-point distances
    dist_points = s[1:]-s[:-1]
    
    # Determine cut points based on large jumps in distance
    if sum(dist_points[:first] >= 10):
        cut_start = s[max(np.where(dist_points[:first] >= 10)[0])+1]
    else: 
        cut_start=0 
    
    if sum(dist_points[last:] >= 10):
        cut_end = s[min(np.where(dist_points[last:] >= 10)[0]) + last]
    else:
        cut_end=len(wh_freqs)-1
    
    # Trim the whistle to the detected limits
    wh_freqs = wh_freqs[cut_start:cut_end]
    wh_times = wh_times[cut_start:cut_end]
    
    return wh_freqs, wh_times

def wh_from_peaks(peak, recording, times, annotation_times, annotation_freqs, annotation_name, alpha=1.25):
    """
    Extract a whistle from a distance peak.

    Parameters
    ----------
    peak : int
        Index of distance minimum peak corresponding to whistle start
    recording : numpy.ndarray
        Frequency vector of the recording
    times : numpy.ndarray
        Time vector of the recording
    annotation_times : numpy.ndarray
        Time vector of the annotation template
    annotation_freqs : numpy.ndarray
        Frequency vector of the annotation template
    annotation_name : str
        Name/identifier of the annotation
    alpha : float, optional
        Scaling factor for whistle duration (default=1.25)

    Returns
    -------
    dict
        Dictionary containing whistle information:
        - whf: frequency values
        - wht: time points
        - onset: start time
        - offset: end time
        - distance: DTW distance
        - normalized distance: distance normalized by length
        - annotation: annotation name
    """
    # Take the peak as t=0
    t0 = times - times[peak]
    # Calculate expected whistle duration based on annotation
    annots_duration = annotation_times[-1]-annotation_times[0]
    wh_duration = alpha*annots_duration
    
    # Extract whistle frequencies and times
    wh_freqs = recording[(t0 <= wh_duration) & (t0 >= 0)]
    ind_times = np.where((t0 <= wh_duration) & (t0 >= 0))[0]
    ind_times = ind_times[ind_times < len(times)]
    
    # Get corresponding time points
    wh_times = times[ind_times]
    
    # Refine whistle boundaries
    wh_freqs, wh_times = get_whistle_limit(annotation_times, annotation_freqs, wh_times, wh_freqs)
    
    # Calculate distances
    dist,_ = fastdtw(annotation_freqs, wh_freqs)
    norm_dist = dist/len(wh_freqs)
    
    # Create whistle dictionary
    wh = {
        'whf': wh_freqs,
        'wht': wh_times,
        'onset': wh_times[0],
        'offset': wh_times[-1],
        'distance': dist,
        'normalized distance': norm_dist,
        'annotation': annotation_name
    }
    return wh

def Extract_Whistles(distance, norm_distance, recording, times, annotation_times, annotation_freqs, annotation_name, threshold=0.2*1e6, n_jobs=-1, alpha=1.25, normalized=False):
    """
    Extract all whistles that are similar to an annotated whistle from a recording using distance metrics.

    Parameters
    ----------
    distance : numpy.ndarray
        Vector of DTW distances between recording and template
    norm_distance : numpy.ndarray
        Vector of normalized DTW distances
    recording : numpy.ndarray
        Frequency vector of the recording
    times : numpy.ndarray
        Time vector of the recording
    annotation_times : numpy.ndarray
        Time vector of the annotation template
    annotation_freqs : numpy.ndarray
        Frequency vector of the annotation template
    annotation_name : str
        Name/identifier of the annotation
    threshold : float, optional
        Distance threshold for whistle detection (default=0.2e6)
    n_jobs : int, optional
        Number of parallel jobs (default=-1)
    alpha : float, optional
        Scaling factor for whistle duration (default=1.25)
    normalized : bool, optional
        Whether to use normalized distances (default=False)

    Returns
    -------
    tuple
        (whistle array, distance vector, peak indices)
        - whistle array: array of whistle dictionaries
        - distance vector: vector of distances used
        - peak indices: indices where whistles were detected
    """
    # Detect whistles using peak detection
    if normalized:
        peaks, props = find_peaks(-norm_distance, height=-threshold, width=20)
    else:
        peaks, props = find_peaks(-distance, height=-threshold, width=20)
    
    # Extract whistles at each peak
    Wh = np.array(list(map(
        lambda peak: wh_from_peaks(peak, recording, times, annotation_times, 
                                 annotation_freqs, annotation_name, alpha=alpha),
        peaks)))
    
    return Wh, distance, peaks


