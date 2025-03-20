# for getting movie data
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import BehaviorOphysExperiment
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

import os
import yaml

import numpy as np
import pandas as pd


# setup for export
def generate_cache(cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)

    return cache


def get_experiment_ids(cache_dir='../data/./visual_behaviour_cache' ,ammount=2):       
    cache=generate_cache(cache_dir)
    experiments = cache.get_ophys_experiment_table()
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
        print("movie 1 saved")

    if not os.path.exists(data_folder + '/natural_movie_three.npy'):
        movie_3 = mv_1_3.get_stimulus_template('natural_movie_three')
        np.save(data_folder + '/natural_movie_three.npy', movie_3)
        print("movie 3 saved")


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