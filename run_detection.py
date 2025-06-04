#!/usr/bin/env python3

from AutomaticExtraction.src.cli.detect_command import detect_command
import argparse

def main():
    """
    Main function to run whistle detection
    """
    parser = argparse.ArgumentParser(description='Detect whistles in audio files')
    
    # Add all the necessary arguments
    parser.add_argument('--model_path', required=True, help='Path to the model file')
    parser.add_argument('--recordings', required=True, help='Path to the recordings directory')
    parser.add_argument('--saving_folder', required=True, help='Directory to save results')
    parser.add_argument('--start_time', type=float, default=0, help='Start time in seconds')
    parser.add_argument('--end_time', type=float, default=None, help='End time in seconds')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--save_p', action='store_true', help='Save positive detections')
    parser.add_argument('--max_workers', type=int, default=8, help='Maximum number of concurrent workers')
    parser.add_argument('--file_list', help='Path to a text file containing specific files to process')
    
    args = parser.parse_args()
    return detect_command(args)

if __name__ == '__main__':
    exit(main()) 