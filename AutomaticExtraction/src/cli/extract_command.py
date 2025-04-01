"""
Extraction Command Module

This module provides command-line interface for extracting whistle contours.
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from ..extraction.contour import extract_whistle_contours_from_file
from ..extraction.clustering import (
    prepare_contour_for_clustering, 
    cluster_whistles, 
    extract_cluster_representatives,
    visualize_clusters
)
from ..utils.helpers import ensure_directory_exists

def setup_extract_parser(subparsers):
    """
    Set up the command-line parser for the extract command.
    
    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers object to add the extract parser to
        
    Returns
    -------
    argparse.ArgumentParser
        The extract command parser
    """
    parser = subparsers.add_parser('extract', help='Extract whistle contours from detection results')
    
    parser.add_argument('--audio_path', required=True, help='Path to the audio file or directory')
    parser.add_argument('--predictions_path', required=True, help='Path to the predictions CSV file or directory')
    parser.add_argument('--output_dir', required=True, help='Directory to save extracted contours')
    parser.add_argument('--cluster', action='store_true', help='Cluster extracted contours')
    parser.add_argument('--distance_threshold', type=float, default=0.3, help='DTW distance threshold for clustering')
    parser.add_argument('--min_samples', type=int, default=2, help='Minimum samples per cluster')
    parser.add_argument('--visualize', action='store_true', help='Visualize clusters')
    
    return parser

def extract_command(args):
    """
    Execute the extract command.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    int
        Exit code (0 for success)
    """
    # Ensure output directory exists
    ensure_directory_exists(args.output_dir)
    
    # Check if audio_path is a file or directory
    if os.path.isfile(args.audio_path):
        # Single file processing
        audio_files = [args.audio_path]
        predictions_files = [args.predictions_path]
    else:
        # Directory processing
        audio_files = [os.path.join(args.audio_path, f) for f in os.listdir(args.audio_path) 
                      if f.lower().endswith('.wav')]
        
        # Find corresponding prediction files
        predictions_files = []
        for audio_file in audio_files:
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            pred_file = os.path.join(args.predictions_path, f"{base_name}.wav_predictions.csv")
            if os.path.exists(pred_file):
                predictions_files.append(pred_file)
            else:
                print(f"Warning: No predictions found for {audio_file}")
                audio_files.remove(audio_file)
    
    # Process each file
    all_contours = []
    file_labels = []
    
    for audio_file, pred_file in zip(audio_files, predictions_files):
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_csv = os.path.join(args.output_dir, f"{base_name}_contours.csv")
        
        print(f"Extracting contours from {base_name}...")
        times, freqs = extract_whistle_contours_from_file(audio_file, pred_file, output_csv)
        
        if len(times) > 0:
            contour = prepare_contour_for_clustering(times, freqs)
            all_contours.append(contour)
            file_labels.append(base_name)
    
    # Cluster contours if requested
    if args.cluster and all_contours:
        print("Clustering whistle contours...")
        labels, distance_matrix = cluster_whistles(
            all_contours, 
            distance_threshold=args.distance_threshold,
            min_samples=args.min_samples
        )
        
        # Save clustering results
        cluster_results = pd.DataFrame({
            'file': file_labels,
            'cluster': labels
        })
        cluster_results.to_csv(os.path.join(args.output_dir, 'cluster_results.csv'), index=False)
        
        # Save distance matrix
        np.save(os.path.join(args.output_dir, 'distance_matrix.npy'), distance_matrix)
        
        # Extract and save representative contours
        representatives = extract_cluster_representatives(all_contours, labels)
        
        for label, contour in representatives.items():
            rep_df = pd.DataFrame({
                'time': contour[:, 0],
                'frequency': contour[:, 1]
            })
            rep_df.to_csv(os.path.join(args.output_dir, f'cluster_{label}_representative.csv'), index=False)
        
        # Visualize clusters if requested
        if args.visualize:
            visualize_clusters(
                all_contours, 
                labels, 
                output_path=os.path.join(args.output_dir, 'whistle_clusters.png')
            )
    
    print(f"Extraction complete. Results saved to {args.output_dir}")
    return 0 