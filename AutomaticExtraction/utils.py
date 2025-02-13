import os
import re
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram, blackman
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def transform_file_name(file_name):
    """
    Transforms a file name from the format 'Exp_DD_MMM_YYYY_HHMM_channel_N' to 'DD_MMM_YY_HHMM_cN'
    """
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

def process_audio_file(file_path, saving_folder="./images", batch_size=50, start_time=0, end_time=None, save=False, wlen=2048,
                       nfft=2048, sliding_w=0.4, cut_low_frequency=3, cut_high_frequency=20, target_width_px=903,
                       target_height_px=677):
    """
    Process audio file to generate spectrograms
    """
    try:
        # Load sound recording
        fs, x = wavfile.read(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")
    
    # Create the saving folder if it doesn't exist
    if save and not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    
    # Calculate the spectrogram parameters
    hop = round(0.8 * wlen)  # window hop size
    win = blackman(wlen, sym=False)

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
        x_w = x[low:low + samples_per_slice]
        
        # Calculate the spectrogram
        f, t, Sxx = spectrogram(x_w, fs, nperseg=wlen, noverlap=hop, nfft=nfft, window=win)
        Sxx = 20 * np.log10(np.abs(Sxx)+1e-14)  # Convert to dB

        # Create the spectrogram plot
        fig, ax = plt.subplots()
        ax.pcolormesh(t, f / 1000, Sxx, cmap='gray')
        ax.set_ylim(cut_low_frequency, cut_high_frequency)

        ax.set_axis_off()  # Turn off axis
        fig.set_size_inches(target_width_px / plt.rcParams['figure.dpi'], target_height_px / plt.rcParams['figure.dpi'])
        
        # Adjust margins
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
      
        # Save the spectrogram as a JPG image without borders
        if save:
            image_name = os.path.join(saving_folder, f"{file_name}-{low/fs}.jpg")
            fig.savefig(image_name, dpi=plt.rcParams['figure.dpi'])  # Save without borders

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

        low += samples_per_slice

    plt.close('all')  # Close all figures to release memory

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

