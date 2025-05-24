# for getting movie data
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import BehaviorOphysExperiment
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

import os
import yaml
import subprocess

import numpy as np
import pandas as pd


# setup for export
def generate_cache(cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)

    return cache


def get_experiment_ids(cache_dir='../data/./visual_behaviour_cache' ,ammount=2, ids=None):       
    cache=generate_cache(cache_dir)
    experiments = cache.get_ophys_experiment_table()
    if ids == None:
        ids = experiments.index[:ammount]
    return cache, ids


### function for creating directory structure
def create_directory_structure(root_folder, base_dir):
    # Define the top-level subdirectories
    os.makedirs(root_folder, exist_ok=True)
    
    subdirectories = ['eye_tracker', 'responses', 'screen', 'treadmill', 'stimuli']
    
    for subdir in subdirectories:
        # Path for the top-level subdirectory
        subdir_path = os.path.join(base_dir, subdir)
        
        # Create the top-level subdirectory
        os.makedirs(subdir_path, exist_ok=True)
        
        # Define the second-level subdirectory: "meta" for all except "screen"
        if subdir == 'screen':
            second_level_subdirs = ['meta', 'data']
        elif subdir == 'stimuli':
            continue
        else:
            second_level_subdirs = ['meta']
        
        for second_level_subdir in second_level_subdirs:
            # Path for the second-level subdirectory
            second_level_path = os.path.join(subdir_path, second_level_subdir)
            
            # Create the second-level subdirectory
            os.makedirs(second_level_path, exist_ok=True)

            if second_level_subdir == 'meta':
                with open(second_level_path+".yml", 'w') as file:
                    yaml.dump({}, file, default_flow_style=False)


# function to add times for blanks
def add_blank_times(df):
    new_rows = []
    
    for i in range(1, len(df)):
        prev_end = df.loc[i - 1, "end_time"]
        curr_start = df.loc[i, "start_time"]

        # Check if there's a gap
        if curr_start > prev_end:
            new_rows.append({"start_time": prev_end, "end_time": curr_start})

    # Append missing rows to the DataFrame
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # Resort DataFrame to maintain order
    return df.sort_values(by="start_time").reset_index(drop=True)


# extra function to save movies since they are not in the templates
def save_movies(data_folder='../data/movies', cache_directory="../data/brain_observatory"):

    # download the necessary data_sets which include the movies
    # already have movie 1 and 2 but we need another set which contains set 3
    
    # Specify the path to the manifest file
    manifest_file = f"{cache_directory}/boc_manifest.json"
    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory)
    
    boc = BrainObservatoryCache(manifest_file = manifest_file)
    mv_1_3 = boc.get_ophys_experiment_data(501940850)
    mv_2 = 0

    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
            
    if not os.path.exists(data_folder + '/natural_movie_one.npy'):
        movie_1 = mv_1_3.get_stimulus_template('natural_movie_one')
        np.save(data_folder + '/natural_movie_one.npy', movie_1)

    if not os.path.exists(data_folder + '/natural_movie_three.npy'):
        movie_3 = mv_1_3.get_stimulus_template('natural_movie_three')
        np.save(data_folder + '/natural_movie_three.npy', movie_3)


def get_cutoff_time(ophys_timestamps, frac=0.5):
    total = len(ophys_timestamps)
    idx = int(total * frac)
    return ophys_timestamps[idx]


def subsample_data(
    presentation,  
    templates,     
    running,      
    dff,          
    events,        
    eye,           
    ophys_timestamps,
    frac=0.5
):
    # Determine bounds
    start_time = presentation.iloc[1]['start_time']  # start at second stimulus
    end_time = get_cutoff_time(ophys_timestamps, frac)

    # Trim presentation
    trimmed_presentation = presentation[
        (presentation['start_time'] >= start_time) & 
        (presentation['start_time'] <= end_time)
    ]

    # Trim ophys timestamps
    trimmed_ophys_times = ophys_timestamps[
        (ophys_timestamps >= start_time) & 
        (ophys_timestamps <= end_time)
    ]

    # Trim dff
    trimmed_dff = dff.copy()
    start_idx = (ophys_timestamps >= start_time).argmax()
    end_idx = start_idx + len(trimmed_ophys_times)

    trimmed_dff['dff'] = trimmed_dff['dff'].apply(
        lambda trace: trace[start_idx:end_idx]
    )

    # Trim events
    trimmed_events = events.copy()
    trimmed_events['events'] = trimmed_events['events'].apply(
        lambda e: e[start_idx:end_idx]
    )
    if 'filtered_events' in trimmed_events.columns:
        trimmed_events['filtered_events'] = trimmed_events['filtered_events'].apply(
            lambda e: e[start_idx:end_idx]
        )

    # Trim running and eye
    trimmed_running = running[
        (running['timestamps'] >= start_time) & 
        (running['timestamps'] <= end_time)
    ]
    trimmed_eye = eye[
        (eye['timestamps'] >= start_time) & 
        (eye['timestamps'] <= end_time)
    ]

    return (
        trimmed_presentation,
        templates, 
        trimmed_running,
        trimmed_dff,
        trimmed_events,
        trimmed_eye,
        trimmed_ophys_times
    )



# function to convert npy vids to mp4
def grayscale_to_rgb_video(grayscale_array, output_path, fps=30, crf=23):
    # Get dimensions
    t, h, w = grayscale_array.shape
    
    # Scale grayscale to 0-255 range if needed
    if grayscale_array.max() <= 1.0:
        grayscale_array = (grayscale_array * 255).astype(np.uint8)
    else:
        grayscale_array = grayscale_array.astype(np.uint8)
    
    # Convert grayscale to RGB
    rgb_array = np.stack([grayscale_array] * 3, axis=-1)
    
    # Set up FFmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',  # Width x height
        '-pix_fmt', 'rgb24',  # Input pixel format
        '-r', str(fps),    # Frame rate
        '-i', '-',         # Input from pipe
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', str(crf),
        f'{output_path}.mp4'
    ]
    
    # Start FFmpeg process
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Prepare all frames at once
    all_frames = rgb_array.tobytes()
    
    # Write all frames to FFmpeg's stdin at once
    process.stdin.write(all_frames)
    
    # Let communicate handle closing stdin
    stdout, stderr = process.communicate()


# small helper function for writing yamls
def write_yaml(data, file_name):
    # Write the row data to the YAML file
    with open(file_name, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


# small helper function for writing data into .mem fles
def write_mem(data, output_dir, filename='data.mem'):
    shape = data.shape
    mem_filename = os.path.join(output_dir, filename)
    mem = np.memmap(mem_filename, dtype='float64', mode='w+', shape=shape)
    mem[:] = data[:]
    mem.flush()