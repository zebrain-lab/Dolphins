"""
Audio Processing Module

This module provides functions for processing audio files and generating spectrograms
for whistle detection.
"""

import os
import numpy as np
import cv2
from scipy.io import wavfile
import matplotlib.pyplot as plt

from .spectrogram import wav_to_spectrogram, create_spectrogram_image

def transform_file_name(file_name):
    """
    Transforms a file name from the format 'Exp_DD_MMM_YYYY_HHMM_channel_N' to 'DD_MMM_YY_HHMM_cN'
    
    Parameters
    ----------
    file_name : str
        Original file name
        
    Returns
    -------
    str or None
        Transformed file name or None if format doesn't match
    """
    import re
    match = re.match(r'Exp_(\d{2})_(\w{3})_(\d{4})_(\d{4})_channel_(\d)', file_name)
    if match:
        day = match.group(1)
        month = match.group(2)
        year = match.group(3)[2:]  # Get last two digits of year
        time = match.group(4)
        channel = match.group(5)
        transformed_name = f"{day}_{month}_{year}_{time}_c{channel}"
        return transformed_name
    else:
        return None

def process_audio_file(file_path, saving_folder="./images", batch_size=50, start_time=0, 
                       end_time=None, save=False, wlen=2048, nfft=2048, sliding_w=0.4, 
                       cut_low_frequency=3, cut_high_frequency=20, target_width_px=903,
                       target_height_px=677):
    """
    Process audio file to generate spectrograms for whistle detection.
    
    Parameters
    ----------
    file_path : str
        Path to the audio file
    saving_folder : str, optional
        Folder to save generated images
    batch_size : int, optional
        Number of spectrograms to generate
    start_time : float, optional
        Start time in seconds
    end_time : float, optional
        End time in seconds, or None for end of file
    save : bool, optional
        Whether to save the generated images
    wlen : int, optional
        Window length for spectrogram
    nfft : int, optional
        FFT size
    sliding_w : float, optional
        Sliding window duration in seconds
    cut_low_frequency : float, optional
        Lower frequency cutoff in kHz
    cut_high_frequency : float, optional
        Upper frequency cutoff in kHz
    target_width_px : int, optional
        Target width of output images
    target_height_px : int, optional
        Target height of output images
        
    Returns
    -------
    list
        List of spectrogram images as numpy arrays
    """
    try:
        # Load sound recording
        fs, x = wavfile.read(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")
    
    # Create the saving folder if it doesn't exist
    if save and not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    
    images = []
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    N = len(x)  # signal length

    if end_time is not None:
        N = min(N, int(end_time * fs))

    low = int(start_time * fs)
    samples_per_slice = int(sliding_w * fs)

    for _ in range(batch_size):
        if low + samples_per_slice > N:  # Check if the slice exceeds the signal length
            break
            
        # Extract the current slice
        x_w = x[low:low + samples_per_slice]
        
        # Calculate the spectrogram
        specgram, freqs, times = wav_to_spectrogram(
            file_path, 
            stride_ms=10.0, 
            window_ms=20.0, 
            max_freq=cut_high_frequency*1000, 
            min_freq=cut_low_frequency*1000,
            cut=(low/fs, (low+samples_per_slice)/fs)
        )
        
        # Create the spectrogram image
        image = create_spectrogram_image(
            specgram, 
            freqs, 
            times, 
            cut_low_freq=cut_low_frequency, 
            cut_high_freq=cut_high_frequency
        )
        
        # Resize to target dimensions
        image = cv2.resize(image, (target_width_px, target_height_px))
        
        # Save the image if requested
        if save:
            image_name = os.path.join(saving_folder, f"{file_name}-{low/fs}.jpg")
            plt.imsave(image_name, image)
        
        images.append(image)
        low += samples_per_slice

    return images 