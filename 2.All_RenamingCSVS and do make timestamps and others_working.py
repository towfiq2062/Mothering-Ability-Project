# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:20:34 2025

@author: mrahman8
"""

import os
import pandas as pd
import chardet
from datetime import datetime

# Define paths
base_dir = r"C:\Users\mrahman8\OneDrive - University of Nebraska-Lincoln\SAU_Drive\0. UNL STUDY\0. Research\0.Current_work\JAN2025\newds153_1-16-25"
output_dir = r"C:\Users\mrahman8\OneDrive - University of Nebraska-Lincoln\SAU_Drive\0. UNL STUDY\0. Research\0.Current_work\JAN2025\csvall"
ds_info_file = r"C:\Users\mrahman8\OneDrive - University of Nebraska-Lincoln\SAU_Drive\0. UNL STUDY\0. Research\0.Current_work\JAN2025\Updated_DS_Sow_info_with_parity.csv"

# Load the DS_Sow_info_with_Mortality_Overlays CSV
ds_info = pd.read_csv(ds_info_file)

# Function to safely read CSV files
def read_csv_safely(file_path):
    encodings = ['utf-8', 'ISO-8859-1', 'latin1']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except Exception:
            continue
    print(f"Failed to read file: {file_path}")
    return None

# Function to process the depth DataFrame
def process_depth_df(depth_df, sow_num):
    def extract_components(filename):
        try:
            parts = filename.split('_')
            date_str = parts[1]  # mmddyyyy
            time_str = parts[2]  # hhmmss.ms
            ms_str = time_str.split('.')[1] if '.' in time_str else '0'

            date = datetime.strptime(date_str, "%m%d%Y").strftime("%m/%d/%Y")
            time = datetime.strptime(time_str[:6], "%H%M%S").strftime("%H:%M:%S")

            return pd.Series({
                'animal_id': sow_num,
                'date': date,
                'hour': time[:2],
                'minute': time[3:5],
                'second': time[6:],
                'ms': ms_str,
                'timestamp': f"{date} {time}.{ms_str}"
            })
        except Exception as e:
            print(f"Error processing filename '{filename}': {e}")
            return pd.Series([None] * 7)

    depth_df[['animal_id', 'date', 'hour', 'minute', 'second', 'ms', 'timestamp']] = depth_df['filename'].apply(extract_components)

    depth_df['belongtobehavior'] = depth_df['predict'].map({
        'SIT': 'sitting',
        'STAND': 'standing',
        'KNEEL': 'kneeling',
        'LOB': 'lying',
        'LOL': 'lying',
        'LOR': 'lying'
    })
    depth_df['posture'] = depth_df['predict'].map({
        'SIT': 'sitting',
        'STAND': 'standing',
        'KNEEL': 'kneeling',
        'LOB': 'lyingonbelly',
        'LOL': 'lyingonleft',
        'LOR': 'lyingonright'
    })
    return depth_df

# Iterate through DS folders
for ds_folder in os.listdir(base_dir):
    ds_folder_path = os.path.join(base_dir, ds_folder)
    if os.path.isdir(ds_folder_path) and ds_folder.startswith('DS'):
        ds_num = ds_folder[2:]
        csv_out_path = os.path.join(ds_folder_path, "csv_out")
        if not os.path.exists(csv_out_path):
            continue

        for sow_folder in os.listdir(csv_out_path):
            sow_folder_path = os.path.join(csv_out_path, sow_folder)
            if os.path.isdir(sow_folder_path):
                sow_num = sow_folder[3:] if sow_folder.startswith("Sow") else sow_folder

                matching_row = ds_info[(ds_info['ds'] == int(ds_num)) & (ds_info['sow_num'] == int(sow_num))]
                if matching_row.empty:
                    continue

                sow_id = matching_row['sow_id'].values[0]
                percent_overlays = matching_row['percent_overlays'].values[0]

                combined_df = pd.DataFrame()

                for date_folder in os.listdir(sow_folder_path):
                    date_folder_path = os.path.join(sow_folder_path, date_folder)
                    depth_folder_path = os.path.join(date_folder_path, 'depth')

                    if os.path.isdir(depth_folder_path):
                        depth_csv_path = os.path.join(depth_folder_path, 'depth.csv')
                        if os.path.exists(depth_csv_path):
                            depth_df = read_csv_safely(depth_csv_path)
                            if depth_df is not None:
                                depth_df = process_depth_df(depth_df, sow_num)
                                combined_df = pd.concat([combined_df, depth_df], ignore_index=True)

                if not combined_df.empty:
                    output_filename = f"{ds_num}_{sow_id}_{percent_overlays}_{sow_num}.csv"
                    output_filepath = os.path.join(output_dir, output_filename)
                    combined_df.to_csv(output_filepath, index=False)
                    print(f"Saved: {output_filepath}")

print("Process completed!")
