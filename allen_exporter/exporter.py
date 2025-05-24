import numpy as np
import os
import yaml
from tqdm import tqdm
import random

from allen_exporter.utils import create_directory_structure, save_movies, write_yaml, write_mem, add_blank_times, get_experiment_ids, grayscale_to_rgb_video, subsample_data

    
def calculate_metrics(stimulus_table, stimulus_templates, running_speed_table,
                      dff_table, event_table, eye_tracking_table, path):

    # Screen: mean and std of pixel values, weighted by image distribution
    stimulus_images = stimulus_table[stimulus_table['image_name'].str.contains('im', case=False, na=False)]
    images_mean = stimulus_templates['warped'].sort_index().apply(np.mean)
    images_std = stimulus_templates['warped'].sort_index().apply(np.std)

    counts = stimulus_images['image_name'].value_counts().sort_index()
    nr_rows = counts.sum()

    weights = counts / nr_rows
    mean_pixel = np.sum(images_mean * weights)
    std_pixel = np.sqrt(np.sum((images_std**2 + (images_mean - mean_pixel)**2) * weights))

    np.save(os.path.join(path, 'screen/meta/means.npy'), mean_pixel)
    np.save(os.path.join(path, 'screen/meta/stds.npy'), std_pixel)

    # Treadmill: average and std of running speed
    mean_speed = running_speed_table['speed'].mean()
    std_speed = running_speed_table['speed'].std()

    np.save(os.path.join(path, 'treadmill/meta/means.npy'), mean_speed)
    np.save(os.path.join(path, 'treadmill/meta/stds.npy'), std_speed)

    # dF/F and events: average and std
    neurons = np.stack(dff_table['dff'].values)

    # Compute mean and std along axis 0 (i.e., for each position 0–11)
    mean_values = np.mean(neurons, axis=1)
    std_values = np.std(neurons, axis=1)

    np.save(os.path.join(path, 'responses/meta/means.npy'), np.array([mean_values]))
    np.save(os.path.join(path, 'responses/meta/stds.npy'), np.array([std_values]))

    # Eye tracker: mean and std of all columns (except timestamps)
    eye_tracking_data = eye_tracking_table.drop(columns='timestamps', errors='ignore')
    mean_values = eye_tracking_data.mean()
    std_values = eye_tracking_data.std()

    np.save(os.path.join(path, 'eye_tracker/meta/means.npy'), mean_values.to_numpy())
    np.save(os.path.join(path, 'eye_tracker/meta/stds.npy'), std_values.to_numpy())

    

# function to save all the images once
def save_images(stimulus_table, stimulus_templates, output_dir, compressed=True):
    images = stimulus_templates['warped']
    idxs = images.index
        
    for i, image in enumerate(images):
        idx = idxs[i]
        npy_path = os.path.join(output_dir, f"{idx}.npy")
        np.save(npy_path, image)

    # save all movies used
    mv_names = stimulus_table[stimulus_table['stimulus_block_name'].str.contains('movie', case=False, na=False)]['stimulus_block_name'].unique()

    if compressed:
        for name in mv_names:
            movie = np.load(f'../data/movies/{name}.npy')
            output_path = os.path.join(output_dir, f"{name}")
            grayscale_to_rgb_video(movie, output_path, fps=30, crf=23)
            
    else:
        for name in mv_names:
            movie = np.load(f'../data/movies/{name}.npy')
            npy_path = os.path.join(output_dir, f"{name}.npy")
            np.save(npy_path, movie)
        

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


# full creation of yml files for the exported stimuli
def stimuli_export(stimuli, stimulus_templates, output_dir, val_rate=0.2,
                   blank_period=0.5, presentation_time=0.25,
                   image_size=[1200, 1900], interleave_value=128, compressed=True):

    frame_rate = float(stimuli['end_frame'].iloc[-1] / stimuli['end_time'].iloc[-1])
    main_yml = os.path.join(output_dir, "meta.yml")
    meta_dict = {
        'modality': 'screen',
        'frame_rate': frame_rate 
    }
    write_yaml(meta_dict, main_yml)

    # Identify image/movie rows for tier assignment
    stimuli = stimuli.copy()
    is_valid_stim = stimuli['image_name'].apply(lambda x: isinstance(x, str) and ('im' in x)) | \
                    ((stimuli['image_name'].apply(lambda x: not isinstance(x, str))) & \
                     (stimuli['stimulus_block_name'].str.contains('movie')))
    
    valid_indices = stimuli[is_valid_stim].index.tolist()
    num_val = int(len(valid_indices) * val_rate)
    val_indices = set(random.sample(valid_indices, num_val))

    timestamps = []
    trial_index = 0
    file_counter = 0
    frame_counter = 0
    was_stimuli = False
    prev_end_time = 0
    
    for idx, row in tqdm(stimuli.iterrows(), desc="processing data", position=0, leave=True):
        yaml_filename = os.path.join(output_dir, f"meta/{file_counter:05}.yml")
        npy_filename = os.path.join(output_dir, f"data/{file_counter:05}.npy")

        image_name = row['image_name']
        first_frame_idx = row["start_frame"]
        end_frame = row["end_frame"]
        is_string = isinstance(image_name, str)
        current_time = row['start_time']

        # Determine tier
        tier = 'val' if idx in val_indices else 'test'

        if is_string and 'im' in image_name:
            if was_stimuli:
                blank_yaml_filename = os.path.join(output_dir, f"meta/{file_counter:05}.yml")
                timestamps.append(prev_end_time)
                yaml_filename, npy_filename, file_counter, frame_counter = write_grey(
                    output_dir, blank_yaml_filename, frame_counter, file_counter, image_size,
                    blank_period, frame_rate
                )

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
            write_yaml(img_data, yaml_filename)
            timestamps.append(current_time)
            was_stimuli = True
            prev_end_time = row['end_time']

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
            timestamps.append(current_time)
            prev_end_time = row['start_time']
            trial_index -= 1
            was_stimuli = is_string

        else:
            if row['movie_frame_index'] != 0:
                continue

            if was_stimuli:
                blank_yaml_filename = os.path.join(output_dir, f"meta/{file_counter:05}.yml")
                timestamps.append(prev_end_time)
                yaml_filename, npy_filename, file_counter, frame_counter = write_grey(
                    output_dir, blank_yaml_filename, frame_counter, file_counter, image_size,
                    blank_period, frame_rate
                )

            movie_name = row["stimulus_block_name"]
            movie = np.load(f'../data/movies/{movie_name}.npy')
            mv_size = movie.shape

            modality = 'encodedvideo' if compressed else 'video'
            file_format = '.mp4' if compressed else '.npy'

            mv_data = {
                'image_name': movie_name,
                'modality': modality,
                'file_format': file_format,
                'tier': tier,
                'stim_type': 'stimulus.Clip',
                'first_frame_idx': frame_counter,
                'trial_index': trial_index,
                'num_frames': mv_size[0],
                'image_size': [mv_size[1], mv_size[2]],
                'pre_blank_period': row['start_time'] - prev_end_time
            }

            write_yaml(mv_data, yaml_filename)
            timestamps.append(current_time)
            for frame_idx in range(1, mv_size[0]):
                timestamps.append(current_time + frame_idx * (2 / frame_rate))

            frame_counter += mv_size[0] - 1
            prev_end_time = row['end_time']
            was_stimuli = True

        frame_counter += 1
        trial_index += 1
        file_counter += 1

    timestamps_array = np.array(timestamps)
    np.save(f'{output_dir}/timestamps.npy', timestamps_array)



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
    write_mem(treadmill_data, output_dir)



def dff_export(dff_table, event_table, timestamps, cells_table, depth, output_dir):

    nr_signals = dff_table['dff'].shape[0]
    d_type = str(dff_table['dff'].iloc[0].dtype)
    nr_rows = dff_table['dff'].iloc[0].shape[0]

    sampling_rate = (dff_table['dff'].values[0].shape / (timestamps[-1] - timestamps[0]))[0]
    
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
        'start_time': float(timestamps[0]), ### first column of timestamps
        'neuron_properties': {
            'cell_motor_coordinates': 'meta/cell_motor_coordinates.npy',
            'unit_ids': 'meta/unit_ids.npy',
            'fields': 'meta/fields.npy'
        }
    }
    
    write_yaml(meta_dict, main_yml)

    # combine all the rows (individual neurons) into a single matrix
    dff_values = np.column_stack(dff_table["dff"].values)
    event_values = np.column_stack(event_table["events"].values)

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
    # ids of cells
    unit_ids = cells_table.index.to_numpy()
    np.save(meta_dir+"unit_ids.npy", unit_ids)
    # ids of cells
    fields = cells_table['height'].to_numpy()
    np.save(meta_dir+"fields.npy", fields)


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
    write_mem(eye_tracking_data, output_dir)

    # save timestamps into meta file
    timestamps = eye_tracking_table['timestamps'].to_numpy()
    np.save(output_dir+"/meta/timestamps.npy", timestamps)


def multi_session_export(ammount, val_rate=0.2, ids=None, compressed=True, root_folder='../data/allen_data', cache_dir='../data/./visual_behaviour_cache',
                         blank_period=0.5, presentation_time=0.25, image_size=[1200, 1900], interleave_value = 128, subsample_frac=1):

    save_movies()
    cache, ids = get_experiment_ids(cache_dir, ammount, ids)
    
    experiments = {}
    for id in ids:
        if id not in experiments.keys():
            print(f'Fetching experiment {id}')
            experiments[id] = cache.get_behavior_ophys_experiment(id)
    
    for id in tqdm(ids, desc=f'Processing Experiment {id}', leave=True):
        base_directory = f'{root_folder}/experiment_{id}'
        create_directory_structure(root_folder, base_directory)
        experiment = experiments[id]

        if subsample_frac != 1:
            (
                presentation,
                templates,
                running,
                dff,
                events,
                eye,
                ophys_times
            ) = subsample_data(
                experiment.stimulus_presentations,
                experiment.stimulus_templates,
                experiment.running_speed,
                experiment.dff_traces,
                experiment.events,
                experiment.eye_tracking,
                experiment.ophys_timestamps,
                subsample_frac
            )

        else:
            presentation = experiment.stimulus_presentations
            templates = experiment.stimulus_templates
            running = experiment.running_speed
            dff = experiment.dff_traces
            events = experiment.events
            ophys_times = experiment.ophys_timestamps
            eye = experiment.eye_tracking
            

        save_images(presentation, templates, f'{base_directory}/stimuli', compressed)
        
        calculate_metrics(presentation, templates,
                          running, dff,
                          events, eye, base_directory)

        stimuli_export(presentation, templates, f'{base_directory}/screen', val_rate,
                       blank_period, presentation_time, image_size, interleave_value, compressed)
        
        treadmill_export(running, f'{base_directory}/treadmill')

        dff_export(dff, events, ophys_times,
                   experiment.cell_specimen_table, experiment.metadata["imaging_depth"], f'{base_directory}/responses')
        
        eye_tracker_export(eye, f'{base_directory}/eye_tracker')

    print('Export completed')
    return cache, ids