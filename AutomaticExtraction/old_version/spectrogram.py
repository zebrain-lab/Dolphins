#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectrogram Analysis Module

This module provides functionality for creating and analyzing spectrograms from audio files.
It includes functions for:
- Computing spectrograms from raw audio samples
- Converting WAV files directly to spectrograms
- Vectorizing spectrograms to extract frequency contours

The implementation uses numpy's FFT and striding operations for efficient computation.
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
    samples : numpy.ndarray
        Waveform samples as a 1D numpy array
    sample_rate : int
        Audio sampling rate in Hz
    stride_ms : float, optional
        Time step between successive windows in milliseconds (default: 5.0)
    window_ms : float, optional
        Size of the analysis window in milliseconds (default: 10.0)
    max_freq : float, optional
        Maximum frequency to include in output (default: np.inf)
    min_freq : float, optional
        Minimum frequency to include in output (default: 0)

    Returns
    -------
    tuple
        Contains:
        - specgram : numpy.ndarray
            2D array containing the spectrogram values (frequency bins × time steps)
        - freqs : numpy.ndarray
            1D array of frequency values corresponding to the frequency bins
        - times : numpy.ndarray
            1D array of time values corresponding to the time steps
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
    Convert a WAV file directly to a spectrogram representation.

    Parameters
    ----------
    recording_path : str
        Path to the WAV file
    stride_ms : float, optional
        Time step between successive windows in milliseconds (default: 5.0)
    window_ms : float, optional
        Size of the analysis window in milliseconds (default: 10.0)
    max_freq : float, optional
        Maximum frequency to include in output (default: np.inf)
    min_freq : float, optional
        Minimum frequency to include in output (default: 0)
    cut : tuple, optional
        Time interval (start_sec, end_sec) to analyze. If None, entire file is processed.

    Returns
    -------
    tuple
        Contains:
        - specgram : numpy.ndarray
            2D array containing the spectrogram values
        - freqs : numpy.ndarray
            1D array of frequency values
        - times : numpy.ndarray
            1D array of time values, adjusted for any cut interval
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
