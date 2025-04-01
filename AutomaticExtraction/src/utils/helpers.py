"""
Helper Utilities

This module provides helper functions used across the application.
"""

import os
import pandas as pd

def read_file_list(file_path):
    """
    Read a list of files from a text file.
    
    Parameters
    ----------
    file_path : str
        Path to the text file
        
    Returns
    -------
    list
        List of file names
    """
    with open(file_path, 'r') as file:
        files = file.read().splitlines()
    return files

def merge_consecutive_windows(df):
    """
    Merge consecutive detection windows in a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'file_name', 'initial_point', 'finish_point', 'confidence'
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with merged consecutive windows
    """
    # Initialize lists to store merged results
    merged_records = []
    current_record = None
    
    for _, row in df.iterrows():
        if current_record is None:
            current_record = {
                'file_name': row['file_name'],
                'initial_point': row['initial_point'],
                'finish_point': row['finish_point'],
                'confidence': row['confidence']
            }
        else:
            # Check if this window is consecutive with the previous one
            if row['initial_point'] == current_record['finish_point']:
                # Update the finish point and average the confidence
                current_record['finish_point'] = row['finish_point']
                current_record['confidence'] = (current_record['confidence'] + row['confidence']) / 2
            else:
                # Save the current record and start a new one
                merged_records.append(current_record)
                current_record = {
                    'file_name': row['file_name'],
                    'initial_point': row['initial_point'],
                    'finish_point': row['finish_point'],
                    'confidence': row['confidence']
                }
    
    # Add the last record if it exists
    if current_record is not None:
        merged_records.append(current_record)
    
    # Create and return a new DataFrame with merged records
    return pd.DataFrame(merged_records)

def ensure_directory_exists(directory_path):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True) 