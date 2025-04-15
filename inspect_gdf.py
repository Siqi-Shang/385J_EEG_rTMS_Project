#!/usr/bin/env python3
"""
load_eeg_gdf.py

A script to load an EEG GDF file using the MNE-Python library.
This script loads the file, prints out basic information (metadata) about the EEG data,
and includes an option to visualize the raw signals.

Usage:
    python load_eeg_gdf.py path/to/your/file.gdf
"""

import argparse
import mne

def load_eeg_gdf(file_path):
    try:
        # Load the GDF file with data preloaded into memory.
        raw = mne.io.read_raw_gdf(file_path, preload=True)
    except Exception as e:
        print(f"Failed to load GDF file: {e}")
        return

    # Print basic information about the data
    print("EEG GDF File Information:")
    print(raw.info)

    # Optionally, plot the raw EEG signals.
    # Uncomment the following line to open an interactive plot.
    # raw.plot()

def main():
    parser = argparse.ArgumentParser(
        description="Load an EEG GDF file and display its metadata using MNE-Python."
    )
    parser.add_argument("--gdf_file", help="Path to the EEG GDF file")
    args = parser.parse_args()

    load_eeg_gdf(args.gdf_file)

if __name__ == '__main__':
    main()
