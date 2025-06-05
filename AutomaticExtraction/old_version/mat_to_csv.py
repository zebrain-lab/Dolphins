import scipy.io
import pandas as pd
import os
from pathlib import Path

def convert_mat_to_csv(mat_file_path, output_dir=None):
    """
    Convert a MAT file containing frequency contours to CSV format.
    Specifically extracts 'whf' (frequency) and 'wht' (time) data.
    
    Args:
        mat_file_path (str): Path to the input MAT file
        output_dir (str, optional): Directory to save the CSV file. If None, 
                                  saves in the same directory as the MAT file.
    
    Returns:
        str: Path to the saved CSV file
    """
    # Load the MAT file
    mat_data = scipy.io.loadmat(mat_file_path)
    
    # Extract frequency and time data
    frequency_data = mat_data['whf'].flatten()  # Flatten in case it's a 2D array
    time_data = mat_data['wht'].flatten()
    
    # Create DataFrame with specific column names
    df = pd.DataFrame({
        'time': time_data,
        'frequency': frequency_data
    })
    
    # Generate output path
    mat_path = Path(mat_file_path)
    if output_dir is None:
        output_dir = mat_path.parent
    
    output_path = Path(output_dir) / f"{mat_path.stem}.csv"
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    return str(output_path)

def batch_convert_mat_to_csv(input_dir, output_dir=None):
    """
    Convert all MAT files in a directory to CSV format.
    
    Args:
        input_dir (str): Directory containing MAT files
        output_dir (str, optional): Directory to save CSV files
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_path
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    mat_files = list(input_path.glob("*.mat"))
    
    for mat_file in mat_files:
        try:
            csv_path = convert_mat_to_csv(str(mat_file), str(output_dir))
            print(f"Converted {mat_file.name} to {Path(csv_path).name}")
        except Exception as e:
            print(f"Error converting {mat_file.name}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert MAT files to CSV format")
    parser.add_argument("input_dir", help="Directory containing MAT files")
    parser.add_argument("--output_dir", help="Directory to save CSV files (optional)")
    
    args = parser.parse_args()
    
    batch_convert_mat_to_csv(args.input_dir, args.output_dir) 