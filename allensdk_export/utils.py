# for getting movie data
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

import os
import yaml

import numpy as np
import pandas as pd


### function for creating directory structure
def create_directory_structure(base_dir):
    # Define the top-level subdirectories
    os.makedirs('../data/allen_data', exist_ok=True)
    
    subdirectories = ['eye_tracker', 'responses', 'screen', 'treadmill']
    
    for subdir in subdirectories:
        # Path for the top-level subdirectory
        subdir_path = os.path.join(base_dir, subdir)
        
        # Create the top-level subdirectory
        os.makedirs(subdir_path, exist_ok=True)
        
        # Define the second-level subdirectory: "meta" for all except "screen"
        if subdir == 'screen':
            second_level_subdirs = ['meta', 'data']
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
            
            #print(f"Created {second_level_path}")
    
    #print(f"Directory structure created at {base_dir}")


# function to get natural movies
def save_movies():

    # download the necessary data_sets which include the movies
    # already have movie 1 and 2 but we need another set which contains set 3

    # Specify the cache directory
    cache_directory = "../data/brain_observatory"
    
    # Specify the path to the manifest file
    manifest_file = f"{cache_directory}/boc_manifest.json"
    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory)
    
    boc = BrainObservatoryCache(manifest_file = manifest_file)
    mv_1_3 = boc.get_ophys_experiment_data(501940850)
    mv_2 = 0

    if not os.path.exists('../data/movies'):
        os.makedirs('../data/movies', exist_ok=True)
            
    if not os.path.exists('../data/movies/natural_movie_one.npy'):
        movie_1 = mv_1_3.get_stimulus_template('natural_movie_one')
        np.save('../data/movies/natural_movie_one.npy', movie_1)
        print("movie 1 saved")

    if not os.path.exists('../data/movies/natural_movie_three.npy'):
        movie_3 = mv_1_3.get_stimulus_template('natural_movie_three')
        np.save('../data/movies/natural_movie_three.npy', movie_3)
        print("movie 3 saved")

    #print("Done saving movies")

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