"""
Detection Command Module

This module provides command-line interface for detecting whistles in audio files.
"""

import os
import argparse
from tqdm import tqdm

from ..detection.model import load_detection_model
from ..detection.predict import process_predict_extract
from ..utils.helpers import ensure_directory_exists, read_file_list

def setup_detect_parser(subparsers):
    """
    Set up the command-line parser for the detect command.
    
    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers object to add the detect parser to
        
    Returns
    -------
    argparse.ArgumentParser
        The detect command parser
    """
    parser = subparsers.add_parser('detect', help='Detect whistles in audio files')
    
    parser.add_argument('--model_path', required=True, help='Path to the model file')
    parser.add_argument('--recordings', required=True, help='Path to the recordings directory')
    parser.add_argument('--saving_folder', required=True, help='Directory to save results')
    parser.add_argument('--start_time', type=float, default=0, help='Start time in seconds')
    parser.add_argument('--end_time', type=float, default=None, help='End time in seconds')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--save_p', action='store_true', help='Save positive detections')
    parser.add_argument('--max_workers', type=int, default=8, help='Maximum number of concurrent workers')
    parser.add_argument('--file_list', help='Path to a text file containing specific files to process')
    
    return parser

def detect_command(args):
    """
    Execute the detect command.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    int
        Exit code (0 for success)
    """
    # Ensure saving directory exists
    ensure_directory_exists(args.saving_folder)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return 1
    
    # Check if recordings directory exists
    if not os.path.exists(args.recordings):
        print(f"Error: Recordings directory not found at {args.recordings}")
        return 1
    
    # Load specific files if provided
    specific_files = None
    if args.file_list and os.path.exists(args.file_list):
        specific_files = read_file_list(args.file_list)
        print(f"Processing {len(specific_files)} files from list")
    
    # Process and predict
    print(f"Starting whistle detection with model: {args.model_path}")
    process_predict_extract(
        recording_folder_path=args.recordings,
        saving_folder=args.saving_folder,
        start_time=args.start_time,
        end_time=args.end_time,
        batch_size=args.batch_size,
        save=False,
        save_p=args.save_p,
        model_path=args.model_path,
        max_workers=args.max_workers,
        specific_files=specific_files
    )
    
    print(f"Detection complete. Results saved to {args.saving_folder}")
    return 0

def main():
    """
    Main function to run the detect command directly.
    """
    parser = argparse.ArgumentParser(description='Detect whistles in audio files')
    subparsers = parser.add_subparsers(dest='command')
    setup_detect_parser(subparsers)
    
    args = parser.parse_args()
    if args.command == 'detect':
        return detect_command(args)
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    exit(main()) 