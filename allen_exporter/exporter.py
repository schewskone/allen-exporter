import numpy as np
import os
from PIL import Image
import yaml
from tqdm import tqdm
import random
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional

from allen_exporter.utils import (
    create_directory_structure, save_movies, write_yaml, write_mem,
    get_experiment_ids, grayscale_to_rgb_video, subsample_data
)
    
def calculate_metrics(
    stimulus_table: pd.DataFrame,
    stimulus_templates: Dict[str, pd.Series],
    running_speed_table: pd.DataFrame,
    dff_table: pd.DataFrame,
    event_table: pd.DataFrame,
    eye_tracking_table: pd.DataFrame,
    path: str
) -> None:

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

    

# function to save all the images once, compressed True, transforms movies into mp4 format
def save_images(
    stimulus_table: pd.DataFrame,
    stimulus_templates: Dict[str, pd.Series],
    output_dir: str,
    compressed: bool = True
) -> None:
    images = stimulus_templates['warped']
    idxs = images.index

    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(images)):
        idx = idxs[i]
        image = images.values[i]

        if compressed:
            if image.max() <= 1.0:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)

            img_pil = Image.fromarray(image_uint8)
            output_path = os.path.join(output_dir, f"{idx}.png")
            img_pil.save(output_path, format='PNG')
        else:
            npy_path = os.path.join(output_dir, f"{idx}.npy")
            np.save(npy_path, image)

    mv_names = stimulus_table[stimulus_table['stimulus_block_name'].str.contains('movie', case=False, na=False)]['stimulus_block_name'].unique()

    for name in mv_names:
        movie = np.load(f'../data/movies/{name}.npy')
        output_path = os.path.join(output_dir, f'{name}')
        if compressed:
            grayscale_to_rgb_video(movie, output_path, fps=30, crf=23)
        else:
            np.save(f'{output_path}.npy', movie)
        

# helper functions for filler grey screens after images and videos
def write_blank(
    yaml_filename: str,
    frame_counter: int,
    image_size: List[int],
    blank_period: float,
    frame_rate: float,
    interleave_value: int = 128
) -> None:
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


# full creation of yml files for the exported stimuli
def stimuli_export(
    stimuli: pd.DataFrame,
    output_dir: str,
    val_rate: float = 0.2,
    blank_period: float = 0.5,
    image_size: List[int] = [1200, 1900],
    interleave_value: int = 128,
    compressed: bool = True
) -> None:

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

    file_format_mv = '.mp4' if compressed else '.npy'
    file_format_img = '.png' if compressed else '.npy'
    
    for idx, row in tqdm(stimuli.iterrows(), desc="processing data", position=0, leave=True):
        yaml_filename = os.path.join(output_dir, f"meta/{file_counter:05}.yml")

        image_name = row['image_name']
        first_frame_idx = row["start_frame"]
        end_frame = row["end_frame"]
        is_string = isinstance(image_name, str)
        current_time = row['start_time']

        # Determine tier
        tier = 'val' if idx in val_indices else 'test'

        if was_stimuli:
            timestamps.append(prev_end_time)
            write_blank(
                yaml_filename, frame_counter, image_size,
                blank_period, frame_rate, interleave_value
            )

            # update variables
            file_counter += 1
            frame_counter += 1
            yaml_filename = os.path.join(output_dir, f"meta/{file_counter:05}.yml")

        if is_string and 'im' in image_name:

            img_data = {
                col: row[col] for col in ['image_name', 'duration', 'stimulus_block_name']} | {
                'image_name': image_name,
                'modality': 'image',
                'file_format': file_format_img,
                'encoded': compressed,
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
            write_blank(
                yaml_filename, frame_counter, image_size,
                row['start_time'] - prev_end_time, frame_rate, interleave_value
            )
            timestamps.append(current_time)
            prev_end_time = row['start_time']
            trial_index -= 1
            was_stimuli = is_string

        else:
            if row['movie_frame_index'] != 0:
                continue

            movie_name = row["stimulus_block_name"]
            movie = np.load(f'../data/movies/{movie_name}.npy')
            mv_size = movie.shape

            modality = 'video' if compressed else 'video'

            mv_data = {
                'image_name': movie_name,
                'modality': modality,
                'file_format': file_format_mv,
                'tier': tier,
                'stim_type': 'stimulus.Clip',
                'first_frame_idx': frame_counter,
                'trial_index': trial_index,
                'num_frames': mv_size[0],
                'encoded': compressed,
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
def treadmill_export(
    speed_table: pd.DataFrame,
    output_dir: str
) -> None:
    
    sampling_rate = speed_table.shape[0] / speed_table["timestamps"].iloc[-1]
    nr_rows = speed_table['speed'].shape[0]
    d_type = str(speed_table['speed'].dtype)
    
    main_yml = os.path.join(output_dir, "meta.yml")
    meta_dict = {
        'dtype': d_type,
        'end_time': float(speed_table['timestamps'].iloc[-1]),
        'is_mem_mapped': True,
        'modality': 'sequence',
        'n_signals': 1,
        'n_timestamps': nr_rows,
        'phase_shift_per_signal': False,
        'sampling_rate': float(sampling_rate),
        'start_time': float(speed_table['timestamps'].iloc[0])
    }
    
    write_yaml(meta_dict, main_yml)

    treadmill_data = np.column_stack((speed_table['timestamps'],speed_table['speed']))
    write_mem(treadmill_data, output_dir)


# export of dff and event response data. Cells_table contains metadata of individual neurons
# event data is not used atm but exported anyway for possible future usage
def dff_export(
    dff_table: pd.DataFrame,
    event_table: pd.DataFrame,
    timestamps: Union[np.ndarray, List[float]],
    cells_table: pd.DataFrame,
    output_dir: str
) -> None:
    
    nr_signals = dff_table['dff'].shape[0]
    d_type = str(dff_table['dff'].iloc[0].dtype)
    nr_rows = dff_table['dff'].iloc[0].shape[0]

    sampling_rate = (dff_table['dff'].values[0].shape / (timestamps[-1] - timestamps[0]))[0]
    
    main_yml = os.path.join(output_dir, "meta.yml")
    meta_dict = {
        'dtype': d_type,
        'end_time': float(timestamps[-1]),
        'is_mem_mapped': True,
        'modality': 'sequence',
        'n_signals': nr_signals,
        'n_timestamps': nr_rows,
        'phase_shift_per_signal': False,
        'sampling_rate': float(sampling_rate),
        'start_time': float(timestamps[0]),
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
    np.save(meta_dir + "cell_ids.npy", dff_table.index.to_numpy())
    np.save(meta_dir + "roi_ids.npy", dff_table["cell_roi_id"].values)
    np.save(meta_dir + "timestamps.npy", timestamps)
    np.save(meta_dir + "cell_motor_coordinates.npy", cells_table[['x', 'y']].to_numpy())
    np.save(meta_dir + "unit_ids.npy", cells_table.index.to_numpy())
    np.save(meta_dir + "fields.npy", cells_table['height'].to_numpy())


def eye_tracker_export(
    eye_tracking_table: pd.DataFrame,
    output_dir: str
) -> None:
        
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


def single_session_export(
    experiment_id: int,
    cache: object,
    val_rate: float = 0.2,
    compressed: bool = True,
    root_folder: str = '../data/allen_data',
    blank_period: float = 0.5,
    image_size: List[int] = [1200, 1900],
    interleave_value: int = 128,
    subsample_frac: float = 1.0
) -> None:
    
    print(f'Processing Experiment {experiment_id}')
    experiment = cache.get_behavior_ophys_experiment(experiment_id)
    base_directory = f'{root_folder}/experiment_{experiment_id}'
    create_directory_structure(root_folder, base_directory)

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

    save_images(presentation, templates, f'{base_directory}/screen/data', compressed)

    calculate_metrics(presentation, templates,
                      running, dff,
                      events, eye, base_directory)

    stimuli_export(presentation, f'{base_directory}/screen', val_rate,
                   blank_period, image_size, interleave_value, compressed)

    treadmill_export(running, f'{base_directory}/treadmill')

    dff_export(dff, events, ophys_times,
               experiment.cell_specimen_table, f'{base_directory}/responses')

    eye_tracker_export(eye, f'{base_directory}/eye_tracker')


def multi_session_export(
    ammount: int,
    val_rate: float = 0.2,
    ids: Optional[List[int]] = None,
    compressed: bool = True,
    root_folder: str = '../data/allen_data',
    cache_dir: str = '../data/./visual_behaviour_cache',
    blank_period: float = 0.5,
    image_size: List[int] = [1200, 1900],
    interleave_value: int = 128,
    subsample_frac: float = 1.0
) -> Tuple[object, List[int]]:
    
    save_movies()
    cache, ids = get_experiment_ids(cache_dir, ammount, ids)

    for experiment_id in tqdm(ids, desc='Processing Experiments', leave=True):
        single_session_export(
            experiment_id,
            cache,
            val_rate=val_rate,
            compressed=compressed,
            root_folder=root_folder,
            blank_period=blank_period,
            image_size=image_size,
            interleave_value=interleave_value,
            subsample_frac=subsample_frac
        )

    print('Export completed')
    return cache, ids
