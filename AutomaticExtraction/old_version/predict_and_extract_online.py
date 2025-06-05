"""
Whistle Detection and Extraction Module

This module handles the detection of dolphin whistles in audio recordings using
a deep learning model and spectrogram analysis.
"""

# =============================================================================
#********************* IMPORTS
# =============================================================================
import warnings
import sys
import os
import pandas as pd # type: ignore
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm 
import cv2 # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
import tensorflow as tf
import concurrent.futures
from utils import *
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ignorer les messages d'information et de débogage de TensorFlow


# =============================================================================
#********************* FUNCTIONS
# =============================================================================

def save_predictions(record_names, positive_initial, positive_finish, class_1_scores, csv_path):
    """Save detection results to a CSV file.
    
    Parameters
    ----------
    record_names : list
        List of file names
    positive_initial : list
        List of start times for positive detections
    positive_finish : list
        List of end times for positive detections
    class_1_scores : list
        List of confidence scores
    csv_path : str
        Path to save the CSV file
    """
    df = pd.DataFrame({
        'file_name': record_names,
        'initial_point': positive_initial,
        'finish_point': positive_finish,
        'confidence': class_1_scores
    })
    df.to_csv(csv_path, index=False)

def process_audio_batch(file_path, batch_duration, start_time, end_time, batch_size, model, save_p, saving_folder_file):
    """Process an audio file in batches and detect whistles.
    
    Parameters
    ----------
    file_path : str
        Path to the audio file
    batch_duration : float
        Duration of each batch in seconds
    start_time : float
        Start time for processing
    end_time : float or None
        End time for processing, or None for end of file
    batch_size : int
        Number of spectrograms to process in each batch
    model : tensorflow.keras.Model
        Model for whistle detection
    save_p : bool
        Whether to save positive detections
    saving_folder_file : str
        Folder to save results
        
    Returns
    -------
    tuple
        (record_names, positive_initial, positive_finish, class_1_scores)
    """
    # Initialize results
    record_names = []
    positive_initial = []
    positive_finish = []
    class_1_scores = []
    
    # Read audio file
    file_name = os.path.basename(file_path)
    fs, x = wavfile.read(file_path)
    N = len(x)

    # Adjust end time if specified
    if end_time is not None:
        N = min(N, int(end_time * fs))

    # Calculate total duration and number of batches
    total_duration = (N / fs) - start_time
    num_batches = int(np.ceil(total_duration / batch_duration))

    # Process each batch
    for batch in tqdm(range(num_batches), desc=f"Processing {file_name}", leave=False):
        batch_start = batch * batch_duration + start_time
        
        # Generate spectrograms for batch
        images = process_audio_file(
            file_path, 
            saving_folder_file, 
            batch_size=batch_size, 
            start_time=batch_start, 
            end_time=end_time
        )
        
        # Skip if no images generated
        if not images:
            continue
            
        # Prepare batch for prediction
        image_batch = []
        time_batch = []
        
        for idx, image in enumerate(images):
            # Calculate time window for spectrogram
            image_start = round(batch_start + idx * 0.4, 2)
            image_end = round(image_start + 0.4, 2)
            
            # Preprocess image
            processed = cv2.resize(image, (224, 224))
            processed = processed / 255.0  # Normalize
            
            image_batch.append(processed)
            time_batch.append((image, image_start, image_end))
        
        # Make predictions on batch
        if image_batch:
            image_batch = np.stack(image_batch)
            predictions = model.predict(image_batch, verbose=0)
            
            # Process positive detections
            for idx, pred in enumerate(predictions):
                if pred[1] > pred[0]:  # Positive class probability > negative
                    image, start, end = time_batch[idx]
                    record_names.append(file_name)
                    positive_initial.append(start)
                    positive_finish.append(end)
                    class_1_scores.append(pred[1])
                    
                    # Save positive detection spectrogram
                    if save_p:
                        save_dir = os.path.join(saving_folder_file, "positive")
                        os.makedirs(save_dir, exist_ok=True)
                        cv2.imwrite(
                            os.path.join(save_dir, f"{start}-{end}.jpg"),
                            image
                        )

    return record_names, positive_initial, positive_finish, class_1_scores

def process_single_file(file_name, recording_folder_path, saving_folder, start_time, end_time, 
                       batch_size, save_p, model, pbar):
    """Process a single audio file for whistle detection.
    
    Parameters
    ----------
    file_name : str
        Name of the file to process
    recording_folder_path : str
        Path to the folder containing recordings
    saving_folder : str
        Folder to save results
    start_time : float
        Start time for processing
    end_time : float or None
        End time for processing
    batch_size : int
        Number of spectrograms to process in each batch
    save_p : bool
        Whether to save positive detections
    model : tensorflow.keras.Model
        Model for whistle detection
    pbar : tqdm.tqdm
        Progress bar
    """
    # Skip non-WAV files or already processed files
    if not file_name.lower().endswith('.wav'):
        print(f"Skipping non-WAV file: {file_name}")
        return
        
    # Setup paths
    base_name = os.path.splitext(file_name)[0]
    file_path = os.path.join(recording_folder_path, file_name)
    saving_folder_file = os.path.join(saving_folder, base_name)
    prediction_file = os.path.join(saving_folder_file, f"{base_name}.wav_predictions.csv")
    
    # Skip if already processed
    if os.path.exists(prediction_file):
        print(f"Skipping already processed file: {file_name}")
        return
        
    # Create output directory
    os.makedirs(saving_folder_file, exist_ok=True)
    
    # Process file
    print(f"Processing: {file_name}")
    batch_duration = batch_size * 0.4
    results = process_audio_batch(
        file_path, batch_duration, start_time, end_time, 
        batch_size, model, save_p, saving_folder_file
    )
    
    # Save results
    save_predictions(*results, prediction_file)
    pbar.update()

def process_predict_extract(recording_folder_path, saving_folder, start_time=0, end_time=None, 
                          batch_size=50, save=False, save_p=True, model_path="models/model_vgg.h5", 
                          max_workers=16, specific_files=None):
    """Process multiple audio files to detect whistles.
    
    Parameters
    ----------
    recording_folder_path : str
        Path to the folder containing recordings
    saving_folder : str
        Folder to save results
    start_time : float, optional
        Start time for processing
    end_time : float or None, optional
        End time for processing
    batch_size : int, optional
        Number of spectrograms to process in each batch
    save : bool, optional
        Whether to save all spectrograms
    save_p : bool, optional
        Whether to save positive detections
    model_path : str, optional
        Path to the model file
    max_workers : int, optional
        Maximum number of concurrent workers
    specific_files : list or None, optional
        List of specific files to process, or None for all files
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Get list of files to process
    if specific_files:
        files = specific_files
    else:
        files = sorted(
            [f for f in os.listdir(recording_folder_path) if f.lower().endswith('.wav')],
            key=lambda x: os.path.getctime(os.path.join(recording_folder_path, x))
        )
    
    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(files), desc="Processing files", position=0) as pbar:
            futures = []
            for file_name in files:
                future = executor.submit(
                    process_single_file,
                    file_name, recording_folder_path, saving_folder,
                    start_time, end_time, batch_size, save_p, model, pbar
                )
                futures.append(future)
                
            # Wait for all files to complete
            concurrent.futures.wait(futures)
