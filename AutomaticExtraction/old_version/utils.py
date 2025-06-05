"""
Utilities Module

This module provides utility functions for audio processing and spectrogram generation.
"""

import os
import re
import numpy as np
import pandas as pd # type: ignore
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.signal.windows import blackman
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Run on CPU for now
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def transform_file_name(file_name):
    """Transform a file name from 'Exp_DD_MMM_YYYY_HHMM_channel_N' to 'DD_MMM_YY_HHMM_cN'.
    
    Parameters
    ----------
    file_name : str
        Input file name
        
    Returns
    -------
    str or None
        Transformed file name, or None if pattern doesn't match
    """
    pattern = r'Exp_(\d{2})_(\w{3})_(\d{4})_(\d{4})_channel_(\d)'
    match = re.match(pattern, file_name)
    
    if match:
        day = match.group(1)
        month = match.group(2)
        year = match.group(3)[2:]  # Get last two digits of year
        time = match.group(4)
        channel = match.group(5)
        return f"{day}_{month}_{year}_{time}_c{channel}"
    return None

def process_audio_file(file_path, saving_folder="./images", batch_size=50, start_time=0, 
                      end_time=None, save=False, wlen=2048, nfft=2048, sliding_w=0.4,
                      cut_low_freq=3, cut_high_freq=20, target_width=903, target_height=677):
    """Generate spectrograms from an audio file.
    
    Parameters
    ----------
    file_path : str
        Path to the audio file
    saving_folder : str, optional
        Directory to save spectrograms
    batch_size : int, optional
        Number of spectrograms to generate
    start_time : float, optional
        Start time in seconds
    end_time : float or None, optional
        End time in seconds, or None for end of file
    save : bool, optional
        Whether to save spectrograms to disk
    wlen : int, optional
        Window length for spectrogram
    nfft : int, optional
        Number of FFT points
    sliding_w : float, optional
        Sliding window duration in seconds
    cut_low_freq : int, optional
        Lower frequency cutoff in kHz
    cut_high_freq : int, optional
        Upper frequency cutoff in kHz
    target_width : int, optional
        Target width of output spectrograms
    target_height : int, optional
        Target height of output spectrograms
        
    Returns
    -------
    list
        List of spectrogram images as numpy arrays
    """
    try:
        # Load audio file
        fs, x = wavfile.read(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Create output directory if saving
    if save and not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    
    # Calculate spectrogram parameters
    hop = round(0.8 * wlen)  # 80% overlap
    win = blackman(wlen, sym=False)
    
    # Initialize results
    images = []
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Calculate signal bounds
    N = len(x)
    if end_time is not None:
        N = min(N, int(end_time * fs))
    
    low = int(start_time * fs)
    samples_per_slice = int(sliding_w * fs)
    
    # Generate spectrograms
    for _ in range(batch_size):
        if low + samples_per_slice > N:
            break
            
        # Extract audio slice
        x_slice = x[low:low + samples_per_slice]
        
        # Calculate spectrogram
        f, t, Sxx = spectrogram(
            x_slice, fs, 
            window=win,
            nperseg=wlen, 
            noverlap=hop, 
            nfft=nfft
        )
        
        # Convert to dB scale
        Sxx = 20 * np.log10(np.abs(Sxx) + 1e-14)
        
        # Create figure
        fig, ax = plt.subplots()
        ax.pcolormesh(t, f / 1000, Sxx, cmap='gray')
        ax.set_ylim(cut_low_freq, cut_high_freq)
        ax.set_axis_off()
        
        # Set figure size
        dpi = plt.rcParams['figure.dpi']
        fig.set_size_inches(target_width/dpi, target_height/dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Save if requested
        if save:
            fig.savefig(
                os.path.join(saving_folder, f"{file_name}-{low/fs}.jpg"),
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0
            )
        
        # Convert to image array
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        image = np.frombuffer(
            renderer.buffer_rgba(),
            dtype=np.uint8
        ).reshape(
            (int(renderer.height), int(renderer.width), 4)
        )[:, :, :3]  # Remove alpha channel
        
        images.append(image)
        plt.close(fig)
        
        # Move to next slice
        low += samples_per_slice
    
    return images

def save_csv(record_names, positive_initial, positive_finish, class_1_scores, csv_path):
    """
    Save detection results to a CSV file
    """
    df = {
        'file_name': record_names,
        'initial_point': positive_initial,
        'finish_point': positive_finish,
        'confidence': class_1_scores
    }

    df = pd.DataFrame(df)
    df.to_csv(csv_path, index=False)
