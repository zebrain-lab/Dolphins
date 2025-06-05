#!/usr/bin/env python3

# usaage : python main.py  --model_path models/model_finetuned_vgg_alexis.h5 --recordings ../../examples/ --saving_folder ../../results/ --batch_size 16 --file_list ../../examples/test_recordings.txt

import argparse
import os
from predict_and_extract_online import process_predict_extract

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Dolphin Whistle Detection')
    
    # Model settings
    parser.add_argument('--model_path', 
                       default="models/model_finetuned_vgg_alexis.h5",
                       help='Path to the detection model')
    
    # Input/Output paths
    parser.add_argument('--recordings', 
                       default="examples/",
                       help='Path to directory containing audio recordings')
    parser.add_argument('--saving_folder', 
                       default='results',
                       help='Directory to save detection results')
    parser.add_argument('--file_list',
                       help='Optional path to text file containing specific files to process')
    
    # Processing parameters
    parser.add_argument('--start_time', 
                       type=float,
                       default=0,
                       help='Start time in seconds for processing')
    parser.add_argument('--end_time', 
                       type=float,
                       default=None,
                       help='End time in seconds for processing (None for full file)')
    parser.add_argument('--batch_size', 
                       type=int,
                       default=32,
                       help='Number of spectrograms to process in each batch')
    parser.add_argument('--max_workers', 
                       type=int,
                       default=8,
                       help='Maximum number of parallel workers')
    
    # Output options
    parser.add_argument('--save_spectrograms',
                       action='store_true',
                       help='Save all generated spectrograms')
    parser.add_argument('--save_positives',
                       action='store_true',
                       help='Save spectrograms of positive detections')
    
    return parser.parse_args()

def read_file_list(file_path):
    """Read list of files from a text file."""
    if not file_path:
        return None
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File list not found: {file_path}")
        
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.saving_folder, exist_ok=True)
    
    # Get list of files to process
    specific_files = read_file_list(args.file_list)
    
    # Run detection
    process_predict_extract(
        recording_folder_path=args.recordings,
        saving_folder=args.saving_folder,
        start_time=args.start_time,
        end_time=args.end_time,
        batch_size=args.batch_size,
        save=args.save_spectrograms,
        save_p=args.save_positives,
        model_path=args.model_path,
        max_workers=args.max_workers,
        specific_files=specific_files
    )

if __name__ == "__main__":
    main()

