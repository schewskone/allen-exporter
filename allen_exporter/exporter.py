import numpy as np
import os
import yaml
from tqdm import tqdm

from allen_exporter.utils import create_directory_structure, save_movies, write_yaml, write_mem, add_blank_times, get_experiment_ids


def calculate_metrics(stimulus_table, stimulus_templates, running_speed_table,
                      dff_table, event_table, eye_tracking_table, path):
    
    # images : average imagecolor over time considering the distribution of images shown
    stimulus_images = stimulus_table[stimulus_table['image_name'].str.contains('im', case=False, na=False)]
    images_mean = stimulus_templates['warped'].sort_index().apply(np.mean) #use .apply here for row wise mean
    counts = stimulus_images['image_name'].value_counts().sort_index()
    nr_rows = sum(counts)
    mean_pixel = str(np.mean(images_mean * (counts / nr_rows)))

    data = {'mean_pixel': mean_pixel}
    write_yaml(data, path+'/screen/meta/mean.yml')

    # treadmill : average running speed
    mean_speed = str(running_speed_table['speed'].mean())
    data = {'mean_speed': mean_speed}
    write_yaml(data, path+'/treadmill/meta/mean.yml')

    # dff : average response
    mean_dff = str(np.mean(dff_table['dff'].mean()))
    mean_event = str(np.mean(event_table['events'].mean()))
    data = {'mean_dff': mean_dff,
            'mean_event': mean_event}
    write_yaml(data, path+'/responses/meta/mean.yml')
    
    # eye tracker : average of everything
    eye_tracking_data = eye_tracking_table.drop('timestamps', axis=1)
    mean_values = eye_tracking_data.mean()
    data = {col: str(mean) for col, mean in mean_values.items()}
    write_yaml(data, path+"/eye_tracker/meta/mean.yml")

    #print('Metrics calculated succesfully')
    

def save_images(stimulus_table, stimulus_templates, output_dir):
    # save all images
    images = stimulus_templates['warped']
    idxs = images.index
        
    for i, image in enumerate(images):
        idx = idxs[i]
        npy_path = os.path.join(output_dir, f"{idx}.npy")
        np.save(npy_path, image)

    # save all movies used
    mv_names = stimulus_table[stimulus_table['stimulus_block_name'].str.contains('movie', case=False, na=False)]['stimulus_block_name'].unique()

    for name in mv_names:
        movie = np.load(f'../data/movies/{name}.npy')
        npy_path = os.path.join(output_dir, f"{name}.npy")
        np.save(npy_path, movie)
        
    #print('Stimuli saved succesfully')
        

# helper functions for filler grey screens after images and videos
def write_grey(output_dir, yaml_filename, frame_counter, file_counter, image_size, blank_period, frame_rate, interleave_value = 128):
    # define grey screen data
    data_grey = {
                    'first_frame_idx': frame_counter,
                    'image_name': 'blank',
                    'image_size': image_size,
                    'modality': 'blank',
                    'interleave_value': interleave_value,
                    'num_frames': frame_rate * blank_period
                }

    write_yaml(data_grey, yaml_filename)

    # update variables

    file_counter += 1
    yaml_filename = os.path.join(output_dir, f"meta/{file_counter:05}.yml")
    npy_filename = os.path.join(output_dir, f"data/{file_counter:05}.npy")

    return yaml_filename, npy_filename, file_counter, frame_counter+1


def stimuli_export(stimuli, stimulus_templates, output_dir, tier='test', frame_rate=60,
                   blank_period=0.5, presentation_time=0.25, image_size=[1200, 1900], interleave_value = 128):

    # make main_yml file with infos for all stimuli
    main_yml = os.path.join(output_dir, "meta.yml")
    meta_dict = {
        'modality': 'screen',
        'frame_rate': frame_rate 
    }
    
    write_yaml(meta_dict, main_yml)

    timestamps = add_blank_times(stimuli[['start_time', 'end_time']])
    timestamps = timestamps['start_time'].to_numpy()
    np.save(f'{output_dir}/timestamps.npy', timestamps)

    trial_index = 0
    file_counter = 0
    frame_counter = 0
    was_stimuli = False
    prev_end_time = 0
    
    for idx, row in tqdm(stimuli.iterrows(), desc="processing data", position=0, leave=True):

        # constructing file name and getting data
        yaml_filename = os.path.join(output_dir, f"meta/{file_counter:05}.yml")
        npy_filename = os.path.join(output_dir, f"data/{file_counter:05}.npy")

        # checking if image_name is string to prevent crashes when checking if 'im' is in image
        image_name = row['image_name']
        first_frame_idx = row["start_frame"]
        end_frame = row["end_frame"]
        is_string = isinstance(image_name, str)

        # stimuli is image
        if is_string and 'im' in image_name:
                        
            # make data for blank yaml if the previous row was image
            if was_stimuli:
                yaml_filename, npy_filename, file_counter, frame_counter = write_grey(output_dir, yaml_filename, frame_counter,
                                                                                     file_counter, image_size,
                                                                                     blank_period, frame_rate)

            
            # get current stimuli template, use this if you want one file for each stimuli
            #img = stimulus_templates.loc[image_name]["warped"]
            #np.save(npy_filename, img)

            img_data = {
                col: row[col] for col in ['image_name', 'duration', 'stimulus_block_name']} | {
                'image_name': image_name,
                'modality': 'image',
                'tier': tier,
                'stim_type': 'stimulus.Frame',
                'first_frame_idx': frame_counter,
                'trial_index': trial_index,
                'num_frames': end_frame - first_frame_idx,
                'image_size': image_size,
                'pre_blank_period': row['start_time'] - prev_end_time
            }
            
            # write yaml with image metadata
            write_yaml(img_data, yaml_filename)
            was_stimuli = True
            prev_end_time = row['end_time']


        # stimuli is grey_screen might have to watchout for image_size and interleave_value
        elif is_string and 'omitted' in image_name or np.isnan(image_name) and 'gray_screen' in row['stimulus_block_name']:
            data_grey = {
                    'first_frame_idx': frame_counter,
                    'image_name': 'blank',
                    'image_size': image_size,
                    'modality': 'blank',
                    'interleave_value': interleave_value,
                    'num_frames': end_frame - first_frame_idx
                }
            write_yaml(data_grey, yaml_filename)
            prev_end_time = row['start_time']
            trial_index -= 1 ### lowering it by one since it increases at the end but shouldn´t
            # if image was omitted a grey screen still appears
            if is_string:
                was_stimuli = True

            else:
                was_stimuli = False

        # stimuli is video
        else:

            # only write movie when first frame appears not for every frame
            if row['movie_frame_index'] != 0:
                continue

            # add grey screen if previous stimuli is image
            if was_stimuli:
                yaml_filename, npy_filename, file_counter, frame_counter = write_grey(output_dir, yaml_filename, frame_counter,
                                                                                     file_counter, image_size,
                                                                                     blank_period, frame_rate)

            movie_name = row["stimulus_block_name"]
            movie = np.load(f'../data/movies/{movie_name}.npy')
            mv_size = movie.shape
            
            mv_data = {
                'image_name': movie_name,
                'modality': 'video',
                'tier': tier,
                'stim_type': 'stimulus.Clip',
                'first_frame_idx': frame_counter,
                'trial_index': trial_index,
                'num_frames': mv_size[0],
                'image_size': [mv_size[1], mv_size[2]],
                'pre_blank_period': row['start_time'] - prev_end_time
            }

            frame_counter += mv_size[0] - 1
            write_yaml(mv_data, yaml_filename)
            prev_end_time = row['end_time']
            
        frame_counter += 1
        trial_index += 1
        file_counter += 1
    
    #print("Visual stimuli sucesfully exported")
    #print(f'Timestamps has len ; {len(timestamps)}')
    #print(f"{file_counter} Files were created!")


# function to export treadmill data
def treadmill_export(speed_table, output_dir):

    sampling_rate = speed_table.shape[0] / speed_table["timestamps"].iloc[-1]
    nr_rows = speed_table['speed'].shape[0]
    d_type = str(speed_table['speed'].dtype)
    
    main_yml = os.path.join(output_dir, "meta.yml")
    meta_dict = {
        'dtype': d_type, ### fix later and get dytpe of column 
        'end_time': float(speed_table['timestamps'].iloc[-1]), ### get value of last time
        'is_mem_mapped': True,
        'modality': 'sequence',
        'n_signals': 1,
        'n_timestamps': nr_rows, ### lenght of columns
        'phase_shift_per_signal': False, ### pretty sure it's always false
        'sampling_rate': float(sampling_rate), ### probably the measurement frequency which is 60 hz, check if we can get this with .metadata
        'start_time': float(speed_table['timestamps'].iloc[0]) ### first column of timestamps
    }
    
    write_yaml(meta_dict, main_yml)

    treadmill_data = np.column_stack((speed_table['timestamps'],speed_table['speed']))
    #print(f'Shape of treadmill_data : {treadmill_data.shape}')
    # speed values to numpy
    write_mem(treadmill_data, output_dir)

    #print("Treadmill data exported succesfully")


def dff_export(dff_table, event_table, timestamps, cells_table, depth, sampling_rate, output_dir):

    nr_signals = dff_table['dff'].shape[0]
    d_type = str(dff_table['dff'].iloc[0].dtype)
    nr_rows = dff_table['dff'].iloc[0].shape[0]
    
    main_yml = os.path.join(output_dir, "meta.yml")
    meta_dict = {
        'dtype': d_type, ### fix later and get dytpe of column 
        'end_time': float(timestamps[-1]), ### get value of last time
        'is_mem_mapped': True,
        'modality': 'sequence',
        'n_signals': nr_signals,
        'n_timestamps': nr_rows, ### lenght of columns
        'phase_shift_per_signal': False, ### pretty sure it's always True here but I can't find any info on how to get the phaseshifts
        'sampling_rate': float(sampling_rate), ### stated in .metadata this is hardcoded and 
        'start_time': float(timestamps[0]) ### first column of timestamps
    }
    
    write_yaml(meta_dict, main_yml)

    # combine all the rows (individual neurons) into a single matrix
    dff_values = np.column_stack(dff_table["dff"].values)
    event_values = np.column_stack(event_table["events"].values)
    #print(f'Shape of dff_values : {dff_values.shape}')
    #print(f'Shape of event_values : {event_values.shape}')

    write_mem(dff_values, output_dir)
    write_mem(event_values, output_dir, 'data_events')

    # write additional info into npy files
    meta_dir = os.path.join(output_dir, "meta/")
    # cell ids
    cell_ids = dff_table.index.to_numpy()
    np.save(meta_dir+"cell_ids.npy", cell_ids)
    # roi id's
    roi_id = dff_table["cell_roi_id"].values
    np.save(meta_dir+"roi_ids.npy", roi_id)
    # timestamps
    np.save(meta_dir+"timestamps.npy", timestamps)
    # coordinates of cells
    motor_coordinates = cells_table[['x','y']].to_numpy()
    np.save(meta_dir+"cell_motor_coordinates.npy", motor_coordinates)

    #print("Responese data exported succesfully")


def eye_tracker_export(eye_tracking_table, output_dir):
    
    nr_rows = eye_tracking_table['timestamps'].shape[0]
    d_type = str(eye_tracking_table['timestamps'].iloc[-1].dtype) ### im just using timestamps here but should be the same anywhere
    end_time = float(eye_tracking_table['timestamps'].iloc[-1])
    sampling_rate = end_time/nr_rows
    n_signals = eye_tracking_table.shape[1]
    
    main_yml = os.path.join(output_dir, "meta.yml")
    meta_dict = {
        'dtype': d_type, ### fix later and get dytpe of column 
        'end_time': end_time, ### get value of last time
        'is_mem_mapped': True,
        'modality': 'sequence',
        'n_signals': n_signals-1, # subtract one because we  do not want timestamps to be included here
        'n_timestamps': nr_rows,
        'phase_shift_per_signal': False, ### pretty sure it's always false here
        'sampling_rate': float(sampling_rate), 
        'start_time': float(eye_tracking_table['timestamps'].iloc[0]) ### first column of timestamps
    }
    
    write_yaml(meta_dict, main_yml)

    eye_tracking_data = eye_tracking_table.drop('timestamps', axis=1).to_numpy()
    #print(f'Shape of eye_tracking_data : {eye_tracking_data.shape}')
    write_mem(eye_tracking_data, output_dir)

    # save timestamps into meta file
    timestamps = eye_tracking_table['timestamps'].to_numpy()
    np.save(output_dir+"/meta/timestamps.npy", timestamps)

    #print("Eyetracker data exported succesfully")



def multi_session_export(ammount, tiers, root_folder='../data/allen_data', cache_dir='../data/./visual_behaviour_cache'):

    save_movies()
    cache, ids = get_experiment_ids(cache_dir, ammount)
    
    experiments = []
    for id in ids:
        print(f'Fetching experiment {id}')
        experiments.append(cache.get_behavior_ophys_experiment(id))
    
    for i, (experiment, tier) in enumerate(tqdm(zip(experiments, tiers), desc=f'Processing Experiment {id}', leave=True)):
        base_directory = f'{root_folder}/experiment_{ids[i]}'
        create_directory_structure(root_folder, base_directory)

        save_images(experiment.stimulus_presentations, experiment.stimulus_templates, f'{base_directory}/stimuli')
        
        calculate_metrics(experiment.stimulus_presentations, experiment.stimulus_templates,
                          experiment.running_speed, experiment.dff_traces,
                          experiment.events, experiment.eye_tracking, base_directory)
        
        stimuli_export(experiment.stimulus_presentations, experiment.stimulus_templates, f'{base_directory}/screen')
        
        treadmill_export(experiment.running_speed, f'{base_directory}/treadmill')

        dff_export(experiment.dff_traces, experiment.events, experiment.ophys_timestamps,
                   experiment.cell_specimen_table, experiment.metadata["imaging_depth"],
                   experiment.metadata["ophys_frame_rate"], f'{base_directory}/responses')
        
        # export of eye_tracking data
        eye_tracker_export(experiment.eye_tracking, f'{base_directory}/eye_tracker')

    print('Export completed')
    return cache, ids