#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whistle Detection and Analysis Module

This module provides functions for detecting and analyzing whistles in spectrograms,
particularly useful for bioacoustic analysis. It includes tools for variance analysis,
whistle detection, and comparison with annotated whistles using Dynamic Time Warping.

Created on Wed Mar 24 10:21:56 2021
@author: faadil
"""

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from joblib import Parallel, delayed
from scipy.signal import find_peaks
from tqdm import tqdm

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


def vectorize_wh_zones(specgram, times, freqs, wh_zone, window_size=5, delta=3):
    """
    Extract frequency-time points from whistle zones in the spectrogram.
    
    Parameters
    ----------
    specgram : numpy.ndarray
        2D spectrogram array (frequency x time)
    times : numpy.ndarray
        Vector of time values corresponding to spectrogram columns
    freqs : numpy.ndarray
        Vector of frequency values corresponding to spectrogram rows
    wh_zone : numpy.ndarray
        Binary array marking whistle zones (1) and non-whistle zones (0)
    window_size : int, optional
        Size of sliding window in time bins (default=5)
    delta : int, optional
        Step size for sliding window (default=3)
        
    Returns
    -------
    tuple
        Two lists containing:
        - whf: Frequencies of detected whistles
        - wht: Corresponding times of detected whistles
    """

    # Progress bar
    pbar = tqdm(total=specgram.shape[1]+1)
    
    # Initialize the vector and the window's indexes
    whf = []
    wht = []
    index_window_min = 0
    index_window_max = window_size
    
    # Loop until the window pass outside the spectrogram
    while index_window_max < specgram.shape[1]:
        
        if sum(wh_zone[index_window_min:index_window_max]) > 0:
            # Select windowed spectrogram 
            current_specgram = specgram[:,index_window_min:index_window_max]
            current_times = times[index_window_min:index_window_max]
            
            # Select only whistle zones on the spectrogram
            current_specgram = current_specgram[:, wh_zone[index_window_min:index_window_max] == 1]
            current_times = current_times[wh_zone[index_window_min:index_window_max] == 1]
            
            # Add all the frequency intensity on the current window
            current_window = np.apply_along_axis(np.sum, 1, current_specgram)
            
            for t in current_times:
                if t not in wht:
                    whf.append(freqs[np.argmax(current_window)])
                    wht.append(t)
                
            
        # Increment the window's indexes
        index_window_min += 1
        index_window_max += 1
        
        # Update progress bar
        pbar.update(1)
    
    # Compute last point
    if wh_zone[-1]:
        # Select windowed spectrogram 
        current_specgram = specgram[:,index_window_min:index_window_max]
        current_times = times[index_window_min:index_window_max]
        
        # Select only whistle zones on the spectrogram
        current_specgram = current_specgram[:, wh_zone[index_window_min:index_window_max] == 1]
        current_times = current_times[wh_zone[index_window_min:index_window_max] == 1]
        
        # Add all the frequency intensity on the current window
        current_window = np.apply_along_axis(np.sum, 1, current_specgram)
    
        for t in current_times:
            if t not in wht:
                whf.append(freqs[np.argmax(current_window)])
                wht.append(t)
    # Update progress bar
    pbar.update(1) 
    
    ## Smooth the whf vector using next and previous point

    # Delta time to select only points that are close in time 
    delta_time = 5*(times[1]-times[0])
    
    ind_time_current = np.where(times == wht[0])[0][0]
    ind_time_next = np.where(times == wht[1])[0][0]
    
    if (ind_time_next-ind_time_current <= delta_time):
        # First point use only next point
        ind_freq_next = np.where(freqs == whf[1])[0][0]
        if ind_freq_next in np.arange(0,delta):
            new_interval = (0, ind_freq_next+delta)
        elif ind_freq_next in np.arange(len(freqs)-delta,len(freqs)):
            new_interval = (ind_freq_next-delta, len(freqs)-1)
        else:   
            new_interval = (ind_freq_next-delta, ind_freq_next+delta)
        
        ind_time_next = np.where(times == wht[1])[0][0]
        
        # Add all the frequency intensity on the current window
        current_window = np.apply_along_axis(np.sum, 1, specgram[:,ind_time_current : ind_time_current + window_size])
        # Boolean vector true on the new interval
        tmp = [0 for i in range(new_interval[0])]+[1 for i in np.arange(new_interval[0], new_interval[1])]+[0 for i in np.arange(new_interval[1], len(current_window))]
        # Assign the max
        whf[0] = freqs[np.argmax(current_window*tmp)]
    
    # Update progress bar
    pbar.update(1)
    
    # Loop from second point to before last point
    for point in np.arange(1, len(whf)-1):
        ind_time_current = np.where(times == wht[point])[0][0]
        ind_time_previous = np.where(times == wht[point-1])[0][0]
        ind_time_next = np.where(times == wht[point+1])[0][0]
        compute_previous = False
        compute_next = False
        
        if (ind_time_current-ind_time_previous <= delta_time):
            ind_previous = np.where(freqs == whf[point-1])[0][0]
            if ind_previous in np.arange(0,delta):
                previous_interval = (0, ind_previous+delta)
            elif ind_previous in np.arange(len(freqs)-delta,len(freqs)):
                previous_interval = (ind_previous-delta, len(freqs)-1)
            else:   
                previous_interval = (ind_previous-delta, ind_previous+delta)
            compute_previous = True
        
        if (ind_time_next-ind_time_current <= delta_time):
            ind_next = np.where(freqs == whf[point+1])[0][0]
            if ind_next in np.arange(0,delta):
                next_interval = (0, ind_next+delta)
            elif ind_next in np.arange(len(freqs)-delta,len(freqs)):
                next_interval = (ind_next-delta, len(freqs)-1)
            else:   
                next_interval = (ind_next-delta, ind_next+delta)
            compute_next = True
        
        if compute_previous :
            if compute_next :
                new_interval = (np.min((previous_interval, next_interval)), 
                                np.max((previous_interval, next_interval)))
            else :
                new_interval = previous_interval
        else :
            if compute_next :
                new_interval = next_interval

        if compute_previous or compute_next :
            # Add all the frequency intensity on the current window
            current_window = np.apply_along_axis(np.sum, 1, specgram[:,ind_time_current : ind_time_current + window_size])
            # Boolean vector true on the new interval
            tmp = [0 for i in range(new_interval[0])]+[1 for i in np.arange(new_interval[0], new_interval[1])]+[0 for i in np.arange(new_interval[1], len(current_window))]
            # Assign the max
            whf[point] = freqs[np.argmax(current_window*tmp)]
            
        # Update progress bar
        pbar.update(1)
        
    # Last point use only previous point
    ind_time_current = np.where(times == wht[-1])[0][0]
    ind_time_previous = np.where(times == wht[point-1])[0][0]
    
    if (ind_time_current-ind_time_previous <= delta_time):
        if ind_time_current == (len(times)-1):
            # If last point of spectrogram, adapt window
            ind_previous = np.where(freqs == whf[-1])[0][0]
            if ind_previous in np.arange(0,delta):
                previous_interval = (0, ind_previous+delta)
            elif ind_previous in np.arange(len(freqs)-delta,len(freqs)):
                previous_interval = (ind_previous-delta, len(freqs)-1)
            else:   
                previous_interval = (ind_previous-delta, ind_previous+delta)
                
            # Add all the frequency intensity on the current window
            current_window = np.apply_along_axis(np.sum, 1, specgram[:,ind_time_current : ind_time_current + window_size])
            # Boolean vector true on the new interval
            tmp = [0 for i in range(new_interval[0])]+[1 for i in np.arange(new_interval[0], new_interval[1])]+[0 for i in np.arange(new_interval[1], len(current_window))]
            # Assign the max
            whf[-1] = freqs[np.argmax(current_window*tmp)]
        else:
            ind_previous = np.where(freqs == whf[-1])[0][0]
            if ind_previous in np.arange(0,delta):
                previous_interval = (0, ind_previous+delta)
            elif ind_previous in np.arange(len(freqs)-delta,len(freqs)):
                previous_interval = (ind_previous-delta, len(freqs)-1)
            else:   
                previous_interval = (ind_previous-delta, ind_previous+delta)
                
            # Add all the frequency intensity on the current window
            current_window = np.apply_along_axis(np.sum, 1, specgram[:,ind_time_current : ind_time_current + window_size])
            # Boolean vector true on the new interval
            tmp = [0 for i in range(new_interval[0])]+[1 for i in np.arange(new_interval[0], new_interval[1])]+[0 for i in np.arange(new_interval[1], len(current_window))]
            # Assign the max
            whf[-1] = freqs[np.argmax(current_window*tmp)]
    
    # Update progress bar
    pbar.update(1)    
    
    return np.array(wht), np.array(whf)


        
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
        dist, _ = fastdtw(annotation_freqs, whf[window[0]:window[1]], dist=euclidean)
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
        dist, _ = fastdtw(annotation_freqs, recording[window[0]:window[1]], dist=euclidean)
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
    first = np.ceil(10/100*len(annotation_freqs))
    last = np.floor(90/100*len(annotation_freqs))
    
    # Compute DTW path between annotation and whistle
    dist, path = fastdtw(annotation_freqs, wh_freqs, dist=euclidean)
    
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
    dist,_ = fastdtw(annotation_freqs, wh_freqs, dist=euclidean)
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
        peaks, props = find_peaks(-norm_distance, height=-threshold, width=30)
    else:
        peaks, props = find_peaks(-distance, height=-threshold, width=30)
    
    # Extract whistles at each peak
    Wh = np.array(list(map(
        lambda peak: wh_from_peaks(peak, recording, times, annotation_times, 
                                 annotation_freqs, annotation_name, alpha=alpha),
        peaks)))
    
    return Wh, distance, peaks


