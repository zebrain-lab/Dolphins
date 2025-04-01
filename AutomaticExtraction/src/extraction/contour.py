"""
Contour Extraction Module

This module provides functionality for extracting frequency contours from spectrograms.
"""

import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2
from ..audio.spectrogram import wav_to_spectrogram
from .dtw import compute_normalized_dtw_distance
from joblib import Parallel, delayed
from scipy.signal import find_peaks
from fastdtw import fastdtw


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

def extract_frequency_contour(specgram, freqs, times, window_size=5, delta=3, min_freq=3000, max_freq=22000, smoothing_window=6):
    """
    Extract the frequency contour from a spectrogram.
    
    Parameters
    ----------
    specgram : numpy.ndarray
        Spectrogram data
    freqs : numpy.ndarray
        Frequency values
    times : numpy.ndarray
        Time values
    window_size : int, optional
        Size of sliding window in time bins (default=5)
    delta : int, optional
        Step size for sliding window (default=3)
    smoothing_window : int, optional
        Size of smoothing window (default=6)
        
    Returns
    -------
    tuple
        Two arrays containing:
        - times: Times of detected frequency peaks
        - peak_freqs: Frequencies of detected peaks
    """
    # n_time_bins = specgram.shape[1]
    # contour_times = []
    # contour_freqs = []
    
    # # Process each time window
    # for t in range(0, n_time_bins - window_size, delta):
    #     # Get the current time window
    #     window = specgram[:, t:t+window_size]
        
    #     # Sum across time to get frequency profile
    #     freq_profile = np.sum(window, axis=1)
        
    #     # Find the peak frequency
    #     if np.max(freq_profile) > 0:
    #         peak_idx = np.argmax(freq_profile)
    #         peak_freq = freqs[peak_idx]
            
    #         # Use the middle time of the window
    #         mid_time = times[t + window_size // 2]
            
    #         contour_times.append(mid_time)
    #         contour_freqs.append(peak_freq)

    contour_freqs = []
    contour_times = []  
    
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
        
        if t not in window_wht:
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
    contour_freqs.extend(window_whf)
    contour_times.extend(window_wht)
    
    # Convert to numpy arrays
    contour_times = np.array(contour_times)
    contour_freqs = np.array(contour_freqs)
    
    # Sort by time
    sort_idx = np.argsort(contour_times)
    contour_times = contour_times[sort_idx]
    contour_freqs = contour_freqs[sort_idx]

    # Apply another smoothing using a rolling median filter
    if len(contour_freqs) > smoothing_window:
        # Use pandas rolling median for smoothing
        df = pd.DataFrame({'freq': contour_freqs})
        smoothed_freqs = df.rolling(window=smoothing_window, center=True).median()
        # Fill NaN values at edges with original values
        smoothed_freqs = smoothed_freqs.fillna(method='bfill').fillna(method='ffill')
        contour_freqs = smoothed_freqs['freq'].values

    return contour_times, contour_freqs

def apply_noise_reduction(specgram, factor=1.5):
    """
    Apply simple noise reduction to a spectrogram.
    
    Parameters
    ----------
    specgram : numpy.ndarray
        Spectrogram data
    factor : float, optional
        Noise reduction factor (default=1.5)
        
    Returns
    -------
    numpy.ndarray
        Noise-reduced spectrogram
    """
    # # Calculate noise floor as the mean of each frequency bin
    # noise_floor = np.mean(specgram, axis=1, keepdims=True)
    
    # # Subtract noise floor from spectrogram
    # reduced = specgram - (noise_floor * factor)
    
    # # Set negative values to zero
    # reduced[reduced < 0] = 0
    
    # Estimate noise level (you can adjust this based on your recordings)
    noise_estimate = np.mean(specgram, axis=1, keepdims=True) * factor

    # Subtract the estimated noise from the spectrogram
    filtered_specgram = np.maximum(specgram - noise_estimate, 0) 

    return filtered_specgram

def vectorize_wh_zones(audio_path, detections_df, 
                       window_size=5, delta=3, 
                       min_freq=3000, max_freq=22000,
                       smoothing_window=6,
                       noise_reduction=False, edge_detection=True):
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
    smoothing_window : int, optional
        Size of smoothing window (default=6)
    noise_reduction : bool, optional
        Whether to apply noise reduction to the spectrogram (default=False)
    edge_detection : bool, optional
        Whether to apply edge detection to the spectrogram (default=True)

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
    pbar = tqdm(total=len(detections_df), desc="Processing whistle zones")
    
    # Process each detected whistle zone
    for _, row in detections_df.iterrows():
        start_time = row['initial_point']
        end_time = row['finish_point']
        
        # Calculate spectrogram for the detected zone
        specgram, freqs, times = wav_to_spectrogram(
            audio_path,
            stride_ms=10.0,
            window_ms=20.0,
            max_freq=max_freq,
            min_freq=min_freq,
            cut=(start_time, end_time)
        )
        
        # Apply noise reduction if requested
        if noise_reduction:
            specgram = apply_noise_reduction(specgram)
        
        # Apply edge detection filter
        if edge_detection:
            specgram = filter_edges_spectrogram(specgram)

        # Extract frequency contour
        zone_times, zone_freqs = extract_frequency_contour(specgram, freqs, times, window_size, delta, min_freq, max_freq, smoothing_window)
        
        # Add to overall results
        if len(zone_times) > 0:
            wht.extend(zone_times)
            whf.extend(zone_freqs)
        
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

def vectorize_whistle_contours_from_file(audio_path, predictions_csv, output_csv=None, 
                                       window_size=5, delta=3, 
                                       min_freq=3000, max_freq=22000, 
                                       smoothing_window=6,
                                       noise_reduction=False, edge_detection=True):
    """
    Extract whistle contours from an audio file based on detection predictions.
    
    Parameters
    ----------
    audio_path : str
        Path to the audio file
    predictions_csv : str
        Path to the CSV file containing whistle predictions
    output_csv : str, optional
        Path to save the extracted contours as CSV
        
    Returns
    -------
    tuple
        Two arrays containing:
        - wht: Times of detected whistles
        - whf: Frequencies of detected whistles
    """
    # Load predictions
    try:
        detections_df = pd.read_csv(predictions_csv)
    except FileNotFoundError:
        print(f"Predictions file not found: {predictions_csv}")
        return np.array([]), np.array([])
    
    # Extract contours
    wht, whf = vectorize_wh_zones(audio_path, detections_df, window_size, 
                                  delta, min_freq, max_freq, 
                                  smoothing_window,
                                  noise_reduction, edge_detection)
    
    # Save to CSV if requested
    if output_csv and len(wht) > 0:
        contour_df = pd.DataFrame({
            'time': wht,
            'frequency': whf
        })
        contour_df.to_csv(output_csv, index=False)
    
    return wht, whf

def split_wh_zones(wht, whf, delta_t=1.0):
    """Split whistle contours into separate segments based on time gaps.
    
    Parameters
    ----------
    wht : numpy.ndarray
        Times of detected whistles
    whf : numpy.ndarray
        Frequencies of detected whistles
    delta_t : float, optional
        Minimum time gap to split whistles (default=1.0)
        
    Returns
    -------
    list
        List of whistle contours, each as a 2D array of [time, frequency] points
    """
    if len(wht) == 0:
        return []
    
    # Sort by time
    sort_idx = np.argsort(wht)
    wht = wht[sort_idx]
    whf = whf[sort_idx]
    
    # List to store separated whistle zones
    split_wh = []
    
    # Initialize first segment
    current_times = [wht[0]]
    current_freqs = [whf[0]]
    
    # Process all points
    for i in range(1, len(wht)):
        # If time gap is small, add to current segment
        if wht[i] - wht[i-1] < delta_t:
            current_times.append(wht[i])
            current_freqs.append(whf[i])
        else:
            # Save current segment and start a new one
            if len(current_times) > 5:  # Only keep segments with enough points
                split_wh.append(np.column_stack((current_times, current_freqs)))
            current_times = [wht[i]]
            current_freqs = [whf[i]]
    
    # Add the last segment
    if len(current_times) > 5:
        split_wh.append(np.column_stack((current_times, current_freqs)))
    
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
        (raw distances array)
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
        annotation_contour = np.column_stack((annotation_times, annotation_freqs))
        whistle_contour = np.column_stack((wht[window[0]:window[1]], whf[window[0]:window[1]]))
        dist = compute_normalized_dtw_distance(annotation_contour, whistle_contour)
        return dist
    
    # Compute distances in parallel
    Distances = Parallel(n_jobs=n_jobs)(delayed(apply_distance)(window) for window in tqdm(windows))
    
    return np.array(Distances)

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

def wh_from_peaks(peak, recording, times, annotation_times, annotation_freqs, alpha=1.25):
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
    
    
    # Create whistle dictionary
    wh = {
        'whf': wh_freqs,
        'wht': wh_times,
        'onset': wh_times[0],
        'offset': wh_times[-1],
    }
    return wh

def Extract_Whistles(distance, recording, times, 
                     annotation_times, annotation_freqs, 
                     threshold=0.15, n_jobs=-1, alpha=1.25, width=10):
    """
    Extract all whistles that are similar to an annotated whistle from a recording using distance metrics.

    Parameters
    ----------
    distance : numpy.ndarray
        Vector of DTW distances between recording and template
    recording : numpy.ndarray
        Frequency vector of the recording
    times : numpy.ndarray
        Time vector of the recording
    annotation_times : numpy.ndarray
        Time vector of the annotation template
    annotation_freqs : numpy.ndarray
        Frequency vector of the annotation template
    threshold : float, optional
        Distance threshold for whistle detection (default=0.15)
    n_jobs : int, optional
        Number of parallel jobs (default=-1)
    alpha : float, optional
        Scaling factor for whistle duration (default=1.25)

    Returns
    -------
    tuple
        (whistle array, distance vector, peak indices)
        - whistle array: array of whistle dictionaries
        - distance vector: vector of distances used
        - peak indices: indices where whistles were detected
    """
    # Detect whistles using peak detection
    peaks, _ = find_peaks(-distance, height=-threshold, width=width)
    
    # Extract whistles at each peak
    # Wh = np.array(list(map(
    #     lambda peak: wh_from_peaks(peak, recording, times, 
    #                                annotation_times, annotation_freqs, alpha=alpha),
    #     peaks)))
    
    Wh = Parallel(n_jobs= n_jobs)(delayed(lambda peak: wh_from_peaks(peak, recording, times, 
                                                                     annotation_times, annotation_freqs, 
                                                                     alpha=alpha))(peak) for peak in peaks)
    
    return Wh, peaks
