import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Define paths
info_file = r"C:\Users\mrahman8\OneDrive - University of Nebraska-Lincoln\SAU_Drive\0. UNL STUDY\0. Research\0.Current_work\JAN2025\Updated_DS_Sow_info_with_parity.csv"
data_folder = r"C:/Users/mrahman8/OneDrive - University of Nebraska-Lincoln/SAU_Drive/0. UNL STUDY/0. Research/0.Current_work/JAN2025/csvall"
output_file = r"C:/Users/mrahman8/OneDrive - University of Nebraska-Lincoln/SAU_Drive/0. UNL STUDY/0. Research/0.Current_work/JAN2025/YOLO_139_processed.csv"

# Load the info file
info_df = pd.read_csv(info_file)
info_df['date'] = pd.to_datetime(info_df['date'], errors='coerce')
info_df = info_df.drop_duplicates(subset=['ds', 'sow_num'])

# List all CSV files in the specified data folder
data_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.csv')]

# Define time segments
time_segments = ["0AM-3AM", "3AM-6AM", "6AM-9AM", "9AM-12PM", "12PM-3PM", "3PM-6PM", "6PM-9PM", "9PM-12AM"]

# Define postures (belongtobehavior)
postures = ["kneeling", "lyingonbelly", "lyingonright", "lyingonleft", "sitting", "standing"]

# Initialize an empty list to store results
all_results = []

def process_file(data_file, ds, sow_num):
    try:
        data_df = pd.read_csv(data_file)
    except Exception as e:
        print(f"Error reading file {data_file}: {e}")
        return None

    # Convert date and ensure valid rows
    data_df['date'] = pd.to_datetime(data_df['date'], errors='coerce')
    if data_df['date'].isna().all():
        print(f"File has no valid date values: {data_file}")
        return None

    # Fetch matching info
    matching_info = info_df[(info_df['ds'] == ds) & (info_df['sow_num'] == sow_num)]
    if matching_info.empty:
        print(f"No matching info for DS: {ds}, Sow Num: {sow_num}")
        return None
    matching_info = matching_info.iloc[0]
    farrowing_date = matching_info['date']

    # Create timestamp
    data_df['timestamp'] = pd.to_datetime(
        data_df['date'].dt.strftime('%Y-%m-%d') + ' ' +
        data_df['hour'].astype(str).str.zfill(2) + ':' +
        data_df['minute'].astype(str).str.zfill(2) + ':' +
        data_df['second'].astype(str).str.zfill(2) + '.' +
        data_df['ms'].astype(str).str.zfill(3),
        format='%Y-%m-%d %H:%M:%S.%f',
        errors='coerce'
    )
    if data_df['timestamp'].isna().any():
        print(f"Timestamp conversion failed for file: {data_file}")
        return None

    # Define time segments
    def time_segment(hour):
        bins = [
            (0, 3, "0AM-3AM"), (3, 6, "3AM-6AM"), (6, 9, "6AM-9AM"), (9, 12, "9AM-12PM"),
            (12, 15, "12PM-3PM"), (15, 18, "3PM-6PM"), (18, 21, "6PM-9PM"), (21, 24, "9PM-12AM")
        ]
        for start, end, label in bins:
            if start <= hour < end:
                return label
        return np.nan

    data_df['time_segment'] = data_df['hour'].apply(time_segment)

    # Group by date and time_segment
    grouped = data_df.groupby(['date', 'time_segment'])

    # Initialize a list to store results for this file
    file_results = []

    for (event_date, time_seg), group in grouped:
        # Skip if fewer than 10 images in the time segment
        if len(group) < 10:
            continue

        # Sort by timestamp
        group = group.sort_values(by='timestamp')

        # Calculate total time (in seconds)
        total_time = (group['timestamp'].max() - group['timestamp'].min()).total_seconds()

        # Calculate missing time (gaps > 60 seconds)
        time_diffs = group['timestamp'].diff().fillna(timedelta(seconds=0)).dt.total_seconds()
        total_missing_time = time_diffs[time_diffs > 60].sum()

        # Ensure total_time + total_missing_time = 10800 seconds
        if total_time + total_missing_time != 10800:
            total_missing_time = 10800 - total_time

        # Calculate total images
        total_images = len(group)

        # Identify events (consecutive postures)
        group['event_id'] = (group['posture'] != group['posture'].shift()).cumsum()

        # Calculate time spent in each posture
        posture_times = group.groupby(['posture', 'event_id']).apply(
            lambda x: (x['timestamp'].max() - x['timestamp'].min()).total_seconds()
        ).reset_index(name='duration')

        # Sum durations for each posture
        posture_times = posture_times.groupby('posture')['duration'].sum().to_dict()

        # Calculate day_cat
        day_diff = (event_date - farrowing_date).days
        day_cat = day_diff

        # Create a dictionary for this group
        result = {
            'event_date': event_date,
            'farrowing_date': farrowing_date,
            'day_cat': day_cat,
            'ds': ds,
            'sow_num': sow_num,
            'time_segment': time_seg,
            'mortality': matching_info['mortality'],
            'sow_id': matching_info['sow_id'],
            'percent_mortality': matching_info['percent_mortality'],
            'percent_overlays': matching_info['percent_overlays'],
            'crate_size': matching_info['crate_size'],
            'crate_category': matching_info['crate_category'],
            'heat_lamps': matching_info['heat_lamps'],
            'parity': matching_info['parity'],
            'total_images': total_images,
            'total_time': total_time,
            'total_missing_time': total_missing_time
        }

        # Add posture times
        for posture in postures:
            result[posture] = posture_times.get(posture, 0)

        file_results.append(result)

    return file_results

# Process all files
for file in data_files:
    filename = os.path.basename(file)
    parts = filename.split('_')
    if len(parts) == 4 and parts[0].isdigit() and parts[3].replace('.csv', '').isdigit():
        ds = int(parts[0])  # Extract ds from the filename
        sow_num = int(parts[3].replace('.csv', ''))  # Extract sow_num from the filename
        print(f"Processing DS: {ds}, Sow Num: {sow_num}, File: {filename}")
        results = process_file(file, ds, sow_num)
        if results:
            all_results.extend(results)

# Convert results to DataFrame
final_df = pd.DataFrame(all_results)

# Save the final output
final_df.to_csv(output_file, index=False)

print(f"Processing complete. Output saved to: {output_file}")