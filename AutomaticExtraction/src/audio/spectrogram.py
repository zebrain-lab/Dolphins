"""
Spectrogram Analysis Module

This module provides functionality for creating and analyzing spectrograms from audio files.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.signal.windows import blackman
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_spectrogram(samples, sample_rate, stride_ms=5.0, 
                        window_ms=10.0, max_freq=np.inf, min_freq=0):
    """
    Compute a spectrogram with consecutive Fourier transforms.

    Parameters
    ----------
    samples : numpy.ndarray
        Waveform samples as a 1D numpy array
    sample_rate : int
        Audio sampling rate in Hz
    stride_ms : float, optional
        Time step between successive windows in milliseconds
    window_ms : float, optional
        Size of the analysis window in milliseconds
    max_freq : float, optional
        Maximum frequency to include in output
    min_freq : float, optional
        Minimum frequency to include in output

    Returns
    -------
    tuple
        (specgram, freqs, times) containing the spectrogram values, frequency bins, and time steps
    """
    # Get number of points for each window and stride
    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)
    
    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, shape=nshape, strides=nstrides)
    
    times = np.linspace(0, (len(samples)+truncate_size)/sample_rate, nshape[1])
    
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

def wav_to_spectrogram(recording_path, stride_ms=5.0, window_ms=10.0, 
                       max_freq=np.inf, min_freq=0, cut=None):
    """
    Convert a WAV file directly to a spectrogram representation.

    Parameters
    ----------
    recording_path : str
        Path to the WAV file
    stride_ms : float, optional
        Time step between successive windows in milliseconds
    window_ms : float, optional
        Size of the analysis window in milliseconds
    max_freq : float, optional
        Maximum frequency to include in output
    min_freq : float, optional
        Minimum frequency to include in output
    cut : tuple, optional
        Time interval (start_sec, end_sec) to analyze. If None, entire file is processed.

    Returns
    -------
    tuple
        (specgram, freqs, times) containing the spectrogram values, frequency bins, and time steps
    """
    sample_rate, samples = wavfile.read(recording_path)
    try:
        samples = samples[:,0]  # Take first channel if stereo
    except IndexError:
        pass  # Already mono
    
    if cut is not None:
        s_start, s_end = cut
        samples_time = np.linspace(0, len(samples)/sample_rate, len(samples))
        samples = samples[(samples_time >= s_start) & (samples_time <= s_end)]
    
    specgram, freqs, times = compute_spectrogram(
        samples, sample_rate, stride_ms=stride_ms, window_ms=window_ms, 
        max_freq=max_freq, min_freq=min_freq
    )
    
    if cut is not None:
        times = times + s_start
    
    return specgram, freqs, times

def create_spectrogram_image(specgram, freqs, times, cut_low_freq=3, cut_high_freq=20):
    """
    Create a spectrogram image from spectrogram data.
    
    Parameters
    ----------
    specgram : numpy.ndarray
        Spectrogram data
    freqs : numpy.ndarray
        Frequency values
    times : numpy.ndarray
        Time values
    cut_low_freq : float, optional
        Lower frequency cutoff in kHz
    cut_high_freq : float, optional
        Upper frequency cutoff in kHz
        
    Returns
    -------
    numpy.ndarray
        RGB image of the spectrogram
    """
    # Convert to dB scale
    specgram_db = 20 * np.log10(np.abs(specgram) + 1e-14)
    
    # Create figure with no margins
    fig, ax = plt.subplots()
    ax.pcolormesh(times, freqs / 1000, specgram_db, cmap='gray')
    ax.set_ylim(cut_low_freq, cut_high_freq)
    ax.set_axis_off()
    
    # Adjust figure size and margins
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Render the figure to an array
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = int(renderer.width), int(renderer.height)
    buf = renderer.buffer_rgba()
    image = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
    
    # Remove alpha channel
    image = image[:, :, :3]
    
    plt.close(fig)
    
    return image 