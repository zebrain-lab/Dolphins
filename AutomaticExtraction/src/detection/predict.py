"""
Prediction Module

This module handles the prediction of whistles in audio files.
"""

import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import concurrent.futures

from ..audio.processing import process_audio_file, transform_file_name
from .model import load_detection_model

def save_csv(record_names, positive_initial, positive_finish, class_1_scores, csv_path):
    """
    Save detection results to a CSV file.
    
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
    df = {
        'file_name': record_names,
        'initial_point': positive_initial,
        'finish_point': positive_finish,
        'confidence': class_1_scores
    }

    df = pd.DataFrame(df)
    df.to_csv(csv_path, index=False)

def process_and_predict(file_path, batch_duration, start_time, end_time, batch_size, model, save_p, saving_folder_file):
    """
    Process an audio file and predict whistles.
    
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
    from scipy.io import wavfile
    
    file_name = os.path.basename(file_path)
    transformed_file_name = transform_file_name(file_name)
    fs, x = wavfile.read(file_path)
    N = len(x)

    if end_time is not None:
        N = min(N, int(end_time * fs))

    total_duration = (N / fs) - start_time
    record_names = []
    positive_initial = []
    positive_finish = []
    class_1_scores = []
    num_batches = int(np.ceil(total_duration / batch_duration))

    for batch in tqdm(range(num_batches), desc=f"Batches for {transformed_file_name}", leave=False, colour='blue'):
        start = batch * batch_duration + start_time
        images = process_audio_file(file_path, saving_folder_file, batch_size=batch_size, start_time=start, end_time=end_time)
        
        if not images:
            continue
            
        # Pre-allocate arrays for batch processing
        batch_size = len(images)
        image_batch = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
        time_batch = []
        
        # Process all images in the batch at once
        for idx, image in enumerate(images):
            image_start_time = round(start + idx * 0.4, 2)
            image_end_time = round(image_start_time + 0.4, 2)
            time_batch.append((image, image_start_time, image_end_time))
            image_batch[idx] = cv2.resize(image, (224, 224)) / 255.0

        # Make predictions on the entire batch at once
        predictions = model.predict(image_batch, verbose=0)
        
        # Process predictions
        positive_indices = predictions[:, 1] > predictions[:, 0]
        for idx in np.where(positive_indices)[0]:
            im_cop, image_start_time, image_end_time = time_batch[idx]
            record_names.append(file_name)
            positive_initial.append(image_start_time)
            positive_finish.append(image_end_time)
            class_1_scores.append(predictions[idx, 1])
            
            if save_p:
                saving_positive = os.path.join(saving_folder_file, "positive")
                if not os.path.exists(saving_positive):
                    os.makedirs(saving_positive)
                image_name = os.path.join(saving_positive, f"{image_start_time}-{image_end_time}.jpg")
                cv2.imwrite(image_name, im_cop)

    return record_names, positive_initial, positive_finish, class_1_scores

def process_predict_extract_worker(file_name, recording_folder_path, saving_folder, start_time, end_time, batch_size, 
                                  save_p, model, pbar):
    """
    Worker function for processing a single file.
    
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
    date_and_channel = os.path.splitext(file_name)[0]
    
    print("Processing:", date_and_channel) 
    saving_folder_file = os.path.join(saving_folder, f"{date_and_channel}")
    os.makedirs(saving_folder_file, exist_ok=True)
    prediction_file_path = os.path.join(saving_folder_file, f"{date_and_channel}.wav_predictions.csv")

    file_path = os.path.join(recording_folder_path, file_name)

    if not file_name.lower().endswith(".wav") or (os.path.exists(prediction_file_path)):
        print(f"Non-audio or already predicted: {file_name}. Skipping processing.")
        return

    batch_duration = batch_size * 0.4
    record_names, positive_initial, positive_finish, class_1_scores = process_and_predict(
        file_path, batch_duration, start_time, end_time, batch_size, model, save_p, saving_folder_file
    )
    save_csv(record_names, positive_initial, positive_finish, class_1_scores, prediction_file_path)    
    pbar.update()

def process_predict_extract(recording_folder_path, saving_folder, start_time=0, end_time=1800, batch_size=50, 
                           save=False, save_p=True, model_path="models/model_vgg.h5", max_workers=16, specific_files=None):
    """
    Process multiple audio files to detect whistles.
    
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
    files = os.listdir(recording_folder_path)
    sorted_files = sorted(files, key=lambda x: os.path.getctime(os.path.join(recording_folder_path, x)), reverse=False)
    
    if specific_files:
        sorted_files = sorted(specific_files, key=lambda x: os.path.getctime(os.path.join(recording_folder_path, x)), reverse=False)

    mask_count = 0  # Counter for filtered files
    model = load_detection_model(model_path)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        with tqdm(total=len(files), desc="Files that are not going to be processed right now: ", position=0, leave=True, colour='green') as pbar:
            for file_name in sorted_files:
                file_path = os.path.join(recording_folder_path, file_name)
                date_and_channel = os.path.splitext(file_name)[0]
                prediction_file_path = os.path.join(saving_folder, f"{date_and_channel}/{date_and_channel}.wav_predictions.csv")
                
                mask = (os.path.isdir(file_path) or 
                        not file_name.lower().endswith('.wav') or 
                        os.path.exists(prediction_file_path))
                    
                if mask:
                    mask_count += 1
                    pbar.update(1)
                    continue
                
                future = executor.submit(
                    process_predict_extract_worker, 
                    file_name, 
                    recording_folder_path, 
                    saving_folder, 
                    start_time, 
                    end_time, 
                    batch_size, 
                    save_p, 
                    model, 
                    pbar
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                future.result() 