#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:21:56 2021

@author: faadil
"""

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from joblib import Parallel, delayed
from scipy.signal import find_peaks

# For graphic interface
import tkinter as tk
from tkinter.ttk import Progressbar
from tqdm import tqdm

# Ignore warnings a specific warning
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)



def variance_vector(specgram, freqs, window_size = 5, n_jobs = -1):
    #### Compute variance at each window    
    
    # Initialize the vector and the window's indexes
    var_wind = np.zeros(specgram.shape[1])
    windows = []
    index_window_min = 0
    index_window_max = window_size
    
    windows.append((index_window_min, index_window_max))
    
    # Loop until the window pass outside the spectrogram
    while index_window_max < specgram.shape[1]:
        
        # Increment the window's indexes
        index_window_min += 1
        index_window_max += 1
        
        # Add to windows list
        windows.append((index_window_min, index_window_max))
    
    # Append same window to the last 4 points
    [windows.append((index_window_min, index_window_max)) for _ in range(4)]
    
    def apply_variance(window):
        # Add all the frequency intensity on the current window
        current_window = np.apply_along_axis(np.sum, 1, specgram[:,window[0]:window[1]])
        
        # Rescale for probabilities
        current_window = current_window/sum(current_window)
        
        # Expectation
        expectation = sum(current_window*freqs)
        expectation_squared = sum(current_window*(freqs**2))
        
        # Variance
        return expectation_squared - expectation**2
    
    
    # Variance vector
    var_wind = Parallel(n_jobs= n_jobs)(delayed(apply_variance)(window) for window in tqdm(windows))
    
    
    # # Initialize the vector and the window's indexes
    # var_wind = np.zeros(specgram.shape[1])
    # index_window_min = 0
    # index_window_max = window_size
    
    # pbar = tqdm(total=specgram.shape[1]+1)
    
    # # Loop until the window pass outside the spectrogram
    # while index_window_max < specgram.shape[1]:
        
    #     # Add all the frequency intensity on the current window
    #     current_window = np.apply_along_axis(np.sum, 1, specgram[:,index_window_min:index_window_max])
        
    #     # rescale for probabilities
    #     current_window = current_window/sum(current_window)
        
    #     # Expectation
    #     expectation = sum(current_window*freqs)
    #     expectation_squared = sum(current_window*(freqs**2))
        
        
    #     # Variance
    #     var_wind[index_window_min:index_window_max] = expectation_squared - expectation**2
        
    #     # Increment the window's indexes
    #     index_window_min += 1
    #     index_window_max += 1
    #     pbar.update(1)
    
    # # Assign last point
    # var_wind[index_window_min:index_window_max] = expectation_squared - expectation**2
    # pbar.update(1)
    
    return np.array(var_wind)


def whistle_zones(var_wind, threshold = 1e6, window_length = 60, selection_percentage = 0.20):
    
    labels = (var_wind <= threshold)*1
    
    
    wh_zone = np.zeros(len(labels), dtype=int)
    
    index_window_min = 0
    index_window_max = window_length
    
    # Loop until the window pass outside the labels vector
    while index_window_max < len(labels):
        # Test if there is a selected percentage of low variance point 
        if sum(labels[index_window_min:index_window_max]) >= selection_percentage*window_length:
            wh_zone[index_window_min:index_window_max] = 1
        
        # Increment the window's indexes
        index_window_min+=1
        index_window_max+=1
    
    # Compute last point
    if sum(labels[index_window_min:index_window_max]) >= selection_percentage*window_length:
        wh_zone[index_window_min:index_window_max] = 1
        
    return wh_zone


def vectorize_wh_zones(specgram, times, freqs, wh_zone, window_size = 5, delta = 3, graph_window=None):
    
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
    # List to store separated whistle zones
    split_wh = []
    
    # Initialization of current whistle zone
    current_whf = [whf[0]]
    current_wht = [wht[0]]
    
    # Progress bar
    # pbar = tqdm(total=len(wht))
    
    # Loop unitl out of whole whistle zone
    i = 1
    while (i < len(wht)):
        # Loop until time points are separated of at least delta_t (default : 1 sec)
        while ((wht[i]-wht[i-1] < delta_t) & (i < len(wht))) :
            current_whf.append(whf[i])
            current_wht.append(wht[i])
            if i == len(wht)-1: break
            i+=1
            # pbar.update(1)
        
        # Add current whistle zone to list
        split_wh.append((current_wht, current_whf))
        
        # Initialize new whistle zone
        current_whf = [whf[i]]
        current_wht = [wht[i]]
        
        # Update index and progress bar 
        i+=1
        # pbar.update(1)
    
    return split_wh
    


def distance_wh_zones(wht, whf, annotation_freqs, annotation_times, alphas = (0.5,1.25), n_jobs=-1, graph_window=None):
    
    # Graphic window display
    if graph_window is not None:
        label = tk.Label(graph_window, text='Computing distance', bg="white")
        label.pack(anchor=tk.W)
        graph_window.update()
        bar = Progressbar(graph_window, orient ="horizontal",length = 400, mode ="determinate")
        bar.pack()
    
    # List to store every window
    windows = []
    
    # get duration of annotation
    len_annots_times = annotation_times[-1] - annotation_times[0]
    
    # First window from t=0 to t= alpha1 (default:1.25) x annotation duration
    ind_min = 0
    ind_max = sum((wht - wht[0]) <= alphas[1]*len_annots_times)
    # Add window to list
    windows.append((ind_min, ind_max))
    # Move left border of window of one frame
    ind_min += 1
    
    # Loop until window is out of spectrogram    
    while (ind_max < len(wht))&(ind_min < len(wht)) :
        # Select rigth border of window such as window length <= alpha1 (default:1.25) x annotation duration 
        ind_max = np.max(np.where(wht <= wht[ind_min]+alphas[1]*len_annots_times))
        
        # Test if window length >= alpha0 (default:0.5) x annotation duration
        if wht[ind_max] - wht[ind_min] >= alphas[0]*len_annots_times:
            # Add window to list
            windows.append((ind_min, ind_max))
            # Move left border of window of one frame
            ind_min += 1
        else :
            # If window length < alpha0 (default:0.5) x annotation duration
            # Get number of points in current window
            n_points = ind_max - ind_min
            # Add same window n_points time to list
            [windows.append((ind_min, ind_max)) for _ in range(n_points+1)]
            # Move left border of n_points+1 frame
            ind_min = ind_max+1
                
        
    # Function for parallelizing
    def apply_distance(window):
        """
        Compute distance on given window.
        """
        dist, _ = fastdtw(annotation_freqs, whf[window[0]:window[1]], dist=euclidean)
        
        norm_dist = dist/(window[1]-window[0])
        
        # # Graphic window display update
        # if graph_window is not None:
        #     bar["value"] += progress
        #     graph_window.update_idletasks()

        
        return dist, norm_dist
    
    # Distance vector
    Distances = Parallel(n_jobs= n_jobs)(delayed(apply_distance)(window) for window in tqdm(windows))
    
    Dist = [d[0] for d in Distances]
    Norm_Dist = [d[1] for d in Distances]
    
    # Graphic window display update
    if graph_window is not None:
        label.forget()
        bar.forget()
        
    return np.array(Dist), np.array(Norm_Dist)
    
    

def distance_per_frame(recording, times, annotation_freqs, annotation_times, n_jobs=-1, graph_window=None):
    """
    Compute distance between every frame (points by points) of the recording 
    and the whistle annotation using DTW.

    Parameters
    ----------
    recording : Frequency vector of the recording.
    times : Time vector of the recording.
    annotation_freqs : Frequency vector of the annotation.
    annotation_times : Time vector of the annotation.
    n_jobs : Number of core to use for computing. -1 means all the cores available,-2 all but one and so on. 
             The default is -1.
    graph_window : optional tkinter frame. Used for plotting a progress bar in a graphic window. Frame should be empty.

    Returns
    -------
    Dist : Distance vector.

    """
    
    # Graphic window display
    if graph_window is not None:
        label = tk.Label(graph_window, text='Computing distance', bg="white")
        label.pack(anchor=tk.W)
        graph_window.update()
        bar = Progressbar(graph_window, orient ="horizontal",length = 400, mode ="determinate")
        bar.pack()
    
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
    
    while (ind_min < len(recording)):
        windows.append((ind_min, ind_max))
        
        ind_min +=1
        ind_max +=1
    
    # # Graphic window display update
    # if graph_window is not None:
    #     progress = 100/len(windows)
    
    # Function for parallelizing
    def apply_distance(window):
        """
        Compute distance on given window.
        """
        dist, _ = fastdtw(annotation_freqs, recording[window[0]:window[1]], dist=euclidean)
        
        # # Graphic window display update
        # if graph_window is not None:
        #     bar["value"] += progress
        #     graph_window.update_idletasks()

        
        return dist
    
    # Distance vector
    Dist = Parallel(n_jobs= n_jobs)(delayed(apply_distance)(window) for window in tqdm(windows))
    
    # Graphic window display update
    if graph_window is not None:
        label.forget()
        bar.forget()
        
    return np.array(Dist)



def get_whistle_end(annotation_times, annotation_freqs, wh_times, wh_freqs, annotation_name):
    """
    Detect whistle end using DTW points association.

    Parameters
    ----------
    annotation_times : 
    annotation_freqs : 
    wh_freqs : 
    wh_times : 

    Returns
    -------

    """
    
    ind_start_end = {'SW_Neo_simple':(28,362), 'SW_Neo_stairs':(38,362),'SW_Luna':(30,379), 'SW_Nana':(100,467), 'SW_Shy':(30,400), 
                     'SW_Nikita_normal':(18,185), 'NSW_2':(37,600), 'NSW_6':(55,446), 'NSW_7':(45,360)}
    
    try :
        first, last = ind_start_end[annotation_name]
    except KeyError:
        first, last = (0,-1)
    
    dist, path = fastdtw(annotation_freqs, wh_freqs, dist=euclidean)
    
    p0 = np.array([t[0] for t in path])
    p1 = np.array([t[1] for t in path])
    
    s = [0]
    for i in range(len(annotation_freqs)):
        s.append(p1[max(np.where(p0==i)[0])])
    
    s = np.array(s)
    
    dist_points = s[1:]-s[:-1]
    
    if sum(dist_points[:first] >= 10) : cut_start = s[max(np.where(dist_points[:first] >= 10)[0])+1]
    else: cut_start=0 
    
    if sum(dist_points[last:] >= 10) : cut_end = s[min(np.where(dist_points[last:] >= 10)[0]) + last]
    else: cut_end=len(wh_freqs)-1
    
    wh_freqs = wh_freqs[cut_start:cut_end]
    wh_times = wh_times[cut_start:cut_end]
    
    return wh_freqs, wh_times



def wh_from_peaks(peak, recording, times, annotation_times, annotation_freqs, annotation_name, alpha=1.25):
    """
    Extract a whistle from a distance peak.

    Parameters
    ----------
    peak : Distance minimum peak corresponding to the beginning of the whistle.
    recording : Frequency vector of the recording.
    times : Time vector of the recording.
    annotation_times : Time vector of the annotation.

    Returns
    -------
    wh : Dictionary composed of the frequency and the time of the whistle {whf, wht}.

    """
    # Take the peak as t=0
    t0 = times - times[peak]
    # alpha (default: 1.25) x Annotated whistle duration 
    annots_duration = annotation_times[-1]-annotation_times[0]
    wh_duration = alpha*annots_duration
    
    # Whistle's frequency
    wh_freqs = recording[(t0 <= wh_duration) & (t0 >= 0)]
    ind_times = np.where((t0 <= wh_duration) & (t0 >= 0))[0]
    ind_times = ind_times[ind_times < len(times)]
    
    # Whistle's time
    wh_times = times[ind_times]
    
    wh_freqs, wh_times = get_whistle_end(annotation_times, annotation_freqs, wh_times, wh_freqs, annotation_name)
    
    dist,_ = fastdtw(annotation_freqs, wh_freqs, dist=euclidean)
    norm_dist = dist/len(wh_freqs)
    
    wh = {'whf':wh_freqs, 'wht':wh_times, 'onset':wh_times[0], 'offset':wh_times[-1], 
          'distance':dist, 'normalized distance':norm_dist, 'annotation':annotation_name}
    return wh

def Extract_Whistles(distance, norm_distance, recording, times, annotation_times, annotation_freqs, annotation_name, threshold = 0.2*1e6, n_jobs=-1, alpha=1.25, normalized=False):
    """
    Extract all the whistles that are similar to an annotated whistle 
    from a vectorized recording while using the distance vector.

    Parameters
    ----------
    distance : Distance vector between the recording and the annotated whistle.
    norm_distance :
    recording : Frequency vector of the recording.
    times : Time vector of the recording.
    annotation_times : Time vector of the annotation.
    threshold : threshold used to select whistles.
    n_jobs : Number of core to use for computing. -1 means all the cores available,-2 all the cores but one and so on. 
             The default is -1.
    normalized : Used normalized distance instead of distance. Optional. The default is False

    Returns
    -------
    Wh : List of all extracted whistles. Whistles saved as dictionaries {whf,wht}.
    peaks : Whistles' beginning indexes.

    """
    
    # Detect whistles
    if normalized:
        peaks, props = find_peaks(-norm_distance, height=-threshold, width=30)
    else:
        peaks, props = find_peaks(-distance, height=-threshold, width=30)
    
    # Extract whistles
    # Wh = Parallel(n_jobs= n_jobs)(delayed(lambda peak: wh_from_peaks(peak, recording, times, annotation_times, alpha=alpha))(peak) for peak in peaks)
    
    Wh = np.array(list(map(lambda peak: wh_from_peaks(peak, recording, times, annotation_times, annotation_freqs, annotation_name, alpha=alpha), peaks)))
    
    # Add distance information to extracted whistles     
    # for i in range(len(Wh)):
    #     # Wh[i]['distance'] = -1*props['peak_heights'][i]
    #     Wh[i]['distance'] = distance[peaks[i]]
    #     Wh[i]['normalized distance'] = norm_distance[peaks[i]]
    
    return Wh, distance, peaks


