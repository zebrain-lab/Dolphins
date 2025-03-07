#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 16:01:32 2021

@author: faadil
"""
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm



def spectrogram(samples, sample_rate, stride_ms = 5.0, 
                window_ms = 10.0, max_freq = np.inf, min_freq = 0):
    """
    Compute a spectrogram with consecutive Fourier transforms.

    Parameters
    ----------
    samples : waveform samples.
    sample_rate : acquisition frequency.
    stride_ms : overlap stride used in ms. The default is 5.0.
    window_ms : window used in ms. The default is 10.0.
    max_freq : frequency maximum. The default is np.inf.
    min_freq : frequency minimum. The default is 0.

    Returns
    -------
    specgram
    freqs 
    times

    """
    
    # Get number of points for each window and stride
    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)
    

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, shape = nshape, strides = nstrides)
    
    times = np.linspace(0, (len(samples)+truncate_size)/sample_rate, nshape[1])
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    # Select the spectogram window  
    ind_max = np.where(freqs <= max_freq)[0][-1] + 1
    ind_min = np.where(freqs >= min_freq)[0][0]
    freqs = freqs[ind_min:ind_max]
    
        
    return fft[ind_min:ind_max, :], freqs, times


def wav_to_spec(recording_path, stride_ms = 5.0, window_ms = 10.0, max_freq = np.inf, min_freq = 0, cut=None):
    """
    Function to get spectrogram from recording path directly. More effecient way to use the memory.

    Parameters
    ----------
    recording_path
    stride_ms : overlap stride used in ms. The default is 5.0.
    window_ms : window used in ms. The default is 10.0.
    max_freq : frequency maximum. The default is np.inf.
    min_freq : frequency minimum. The default is 0.
    cut : optionnal. Used for selecting a portion of the recording.
          Use a tuple (s_start, s_end) where s_start and s_end are 
          the beginning and the end of the section in seconds.

    Returns
    -------
    specgram
    freqs 
    times

    """
    
    sample_rate, samples = wavfile.read(recording_path)
    try:
        samples = samples[:,0]
    except IndexError:
        pass
    
    if cut is not None:
        s_start, s_end = cut
        samples_time = np.linspace(0, len(samples)/sample_rate, len(samples))
        samples = samples[(samples_time >= s_start)&(samples_time <= s_end)]
    
    specgram, freqs, times = spectrogram(samples, sample_rate, stride_ms=stride_ms, window_ms=window_ms, 
                                         max_freq=max_freq, min_freq=min_freq)
    
    if cut is not None:
        times = times +s_start
    
    return specgram, freqs, times



def Vectorize_Spectrogram(specgram, freqs, window_size = 4, delta = 3):
    """
    Compute a frequency signal into a vector from a spectrogram using the maximum 
    amplitude on a small window.
    
    Secondly, the signal is smoothed by computing the maximum on a selected interval
    depending on the previous and the next point.

    Parameters
    ----------
    specgram
    freqs
    window_size : Size of the window used to get the max. The default is 4.
    delta : delta considered to compute the interval for each point. the default is 3.
    graph_window : optional tkinter frame. Used for plotting a progress bar in a graphic window. Frame should be empty.

    Returns
    -------
    vect
    
    """    
    
    #### Assign max on each window to the vector
    
    # Initialize the vector and the window's indexes for first loop
    vect = np.zeros(specgram.shape[1])
    index_window_min = 0
    index_window_max = window_size
    # Progress bar using tqdm
    pbar = tqdm(total = 2*(specgram.shape[1]+1))
    
    # Loop until the window pass outside the spectrogram
    while index_window_max < specgram.shape[1]:
        
        # Add all the frequency intensity on the current window
        current_window = np.apply_along_axis(sum, 1, 
                                             specgram[:,index_window_min:index_window_max])
        
        # Select the frequency maximum on the window
        # and assign it to the vector
        vect[index_window_min:index_window_max] = freqs[np.argmax(current_window)]
        
        # Increment the window's indexes
        index_window_min += 1
        index_window_max += 1
        # Update progress bar
        pbar.update(1)
    
    # Assign last point
    vect[index_window_min:index_window_max] = freqs[np.argmax(current_window)]
    # Update progress bar
    pbar.update(1)
    
    #### Smooth the signal using previous and next point
    
    # Initialize for second loop
    index_window_min = 0
    index_window_max = window_size
    
    # First point use only next point
    ind_next = np.where(freqs == vect[1])[0][0]
    if ind_next in np.arange(0,delta):
        new_interval = (0, ind_next+delta)
    elif ind_next in np.arange(len(freqs)-delta,len(freqs)):
        new_interval = (ind_next-delta, len(freqs)-1)
    else:   
        new_interval = (ind_next-delta, ind_next+delta)
        
    # Add all the frequency intensity on the current window
    current_window = np.apply_along_axis(np.sum, 1, specgram[:,index_window_min:index_window_max])
    # Boolean vector true on the new interval
    tmp = [0 for i in range(new_interval[0])]+[1 for i in np.arange(new_interval[0], new_interval[1])]+[0 for i in np.arange(new_interval[1], len(current_window))]
    # Assign the max
    vect[0] = freqs[np.argmax(current_window*tmp)]
    # Update progress bar
    pbar.update(1)
    
    
    # Loop from second point to before last point
    for point in np.arange(1, len(vect)-1):
        ind_previous = np.where(freqs == vect[point-1])[0][0]
        if ind_previous in np.arange(0,delta):
            previous_interval = (0, ind_previous+delta)
        elif ind_previous in np.arange(len(freqs)-delta,len(freqs)):
            previous_interval = (ind_previous-delta, len(freqs)-1)
        else:   
            previous_interval = (ind_previous-delta, ind_previous+delta)
            
        ind_next = np.where(freqs == vect[point+1])[0][0]
        if ind_next in np.arange(0,delta):
            next_interval = (0, ind_next+delta)
        elif ind_next in np.arange(len(freqs)-delta,len(freqs)):
            next_interval = (ind_next-delta, len(freqs)-1)
        else:   
            next_interval = (ind_next-delta, ind_next+delta)
        
        new_interval = (np.min((previous_interval, next_interval)), 
                        np.max((previous_interval, next_interval)))
        
        # Add all the frequency intensity on the current window
        current_window = np.apply_along_axis(np.sum, 1, specgram[:,index_window_min:index_window_max])
        # Boolean vector true on the new interval
        tmp = [0 for i in range(new_interval[0])]+[1 for i in np.arange(new_interval[0], new_interval[1])]+[0 for i in np.arange(new_interval[1], len(current_window))]
        # Assign the max
        vect[point] = freqs[np.argmax(current_window*tmp)]
        
        # Increment the window's indexes
        index_window_min += 1
        index_window_max += 1
        # Update progress bar
        pbar.update(1)
        
    # Last point use only previous point
    ind_previous = np.where(freqs == vect[-1])[0][0]
    if ind_previous in np.arange(0,delta):
        previous_interval = (0, ind_previous+delta)
    elif ind_previous in np.arange(len(freqs)-delta,len(freqs)):
        previous_interval = (ind_previous-delta, len(freqs)-1)
    else:   
        previous_interval = (ind_previous-delta, ind_previous+delta)
        
    # Add all the frequency intensity on the current window
    current_window = np.apply_along_axis(np.sum, 1, specgram[:,index_window_min:index_window_max])
    # Boolean vector true on the new interval
    tmp = [0 for i in range(new_interval[0])]+[1 for i in np.arange(new_interval[0], new_interval[1])]+[0 for i in np.arange(new_interval[1], len(current_window))]
    # Assign the max
    vect[-1] = freqs[np.argmax(current_window*tmp)]
    pbar.update(1)
    
    return vect
