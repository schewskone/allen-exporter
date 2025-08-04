# for getting movie data
import os
import subprocess
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import \
    BehaviorOphysExperiment
from allensdk.brain_observatory.behavior.behavior_project_cache import \
    VisualBehaviorOphysProjectCache
from allensdk.core.brain_observatory_cache import BrainObservatoryCache


# generates cache from which we later on load the nwb files
def generate_cache(cache_dir: str) -> VisualBehaviorOphysProjectCache:
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)
    return cache


def get_experiment_ids(
    cache_dir: str = "data/./visual_behaviour_cache",
    amount: int = 2,
    ids: Optional[List[int]] = None,
) -> Tuple[VisualBehaviorOphysProjectCache, List[int]]:
    cache = generate_cache(cache_dir)
    experiments = cache.get_ophys_experiment_table()
    if ids is None:
        ids = experiments.index[:amount].tolist()
        warnings.warn(
            f"No experiment IDs specified. Defaulting to first {amount} IDs: {ids}",
            UserWarning,
        )
    return cache, ids


def create_directory_structure(root_folder: str, base_dir: str) -> None:
    subdirectories = ["eye_tracker", "responses", "screen", "treadmill", "stimuli"]
    os.makedirs(root_folder, exist_ok=True)

    for subdir in subdirectories:
        subdir_path = os.path.join(base_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)

        if subdir == "screen":
            second_level_subdirs = ["meta", "data"]
        elif subdir == "stimuli":
            continue
        else:
            second_level_subdirs = ["meta"]

        for second_level_subdir in second_level_subdirs:
            second_level_path = os.path.join(subdir_path, second_level_subdir)
            os.makedirs(second_level_path, exist_ok=True)
            if second_level_subdir == "meta":
                with open(second_level_path + ".yml", "w") as file:
                    yaml.dump({}, file, default_flow_style=False)


def save_movies(
    data_folder: str = "data/movies", cache_directory: str = "data/brain_observatory"
) -> None:
    manifest_file = f"{cache_directory}/boc_manifest.json"
    os.makedirs(cache_directory, exist_ok=True)

    boc = BrainObservatoryCache(manifest_file=manifest_file)
    mv_1_3 = boc.get_ophys_experiment_data(501940850)
    mv_2 = boc.get_ophys_experiment_data(577225417)

    os.makedirs(data_folder, exist_ok=True)

    for name in ["natural_movie_one", "natural_movie_three"]:
        path = os.path.join(data_folder, f"{name}.npy")
        if not os.path.exists(path):
            movie = mv_1_3.get_stimulus_template(name)
            np.save(path, movie)

    if not os.path.exists(data_folder + "/natural_movie_two.npy"):
        movie_2 = mv_2.get_stimulus_template("natural_movie_two")
        np.save(data_folder + "/natural_movie_two.npy", movie_2)


# cutoff time refers to the percentage of timepoints we want to export from a single experiment in case we want to subsample experiments
def get_cutoff_time(ophys_timestamps: np.ndarray, frac: float = 0.5) -> float:
    total = len(ophys_timestamps)
    idx = int(total * frac)
    return ophys_timestamps[idx]


def subsample_data(
    presentation: pd.DataFrame,
    templates: Union[np.ndarray, Dict],
    running: pd.DataFrame,
    dff: pd.DataFrame,
    events: pd.DataFrame,
    eye: pd.DataFrame,
    ophys_timestamps: np.ndarray,
    frac: float = 0.5,
) -> Tuple[
    pd.DataFrame,
    Union[np.ndarray, Dict],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
]:
    start_time = presentation.iloc[1]["start_time"]
    end_time = get_cutoff_time(ophys_timestamps, frac)

    trimmed_presentation = presentation[
        (presentation["start_time"] >= start_time)
        & (presentation["start_time"] <= end_time)
    ]

    trimmed_ophys_times = ophys_timestamps[
        (ophys_timestamps >= start_time) & (ophys_timestamps <= end_time)
    ]

    trimmed_dff = dff.copy()
    start_idx = (ophys_timestamps >= start_time).argmax()
    end_idx = start_idx + len(trimmed_ophys_times)

    trimmed_dff["dff"] = trimmed_dff["dff"].apply(
        lambda trace: trace[start_idx:end_idx]
    )

    trimmed_events = events.copy()
    trimmed_events["events"] = trimmed_events["events"].apply(
        lambda e: e[start_idx:end_idx]
    )
    if "filtered_events" in trimmed_events.columns:
        trimmed_events["filtered_events"] = trimmed_events["filtered_events"].apply(
            lambda e: e[start_idx:end_idx]
        )

    trimmed_running = running[
        (running["timestamps"] >= start_time) & (running["timestamps"] <= end_time)
    ]
    trimmed_eye = eye[
        (eye["timestamps"] >= start_time) & (eye["timestamps"] <= end_time)
    ]

    return (
        trimmed_presentation,
        templates,
        trimmed_running,
        trimmed_dff,
        trimmed_events,
        trimmed_eye,
        trimmed_ophys_times,
    )


# Convert a 3D grayscale video array to an RGB video and save as an MP4 file using H.264 compression.
def grayscale_to_rgb_video(
    grayscale_array: np.ndarray, output_path: str, fps: int = 30, crf: int = 23
) -> None:
    t, h, w = grayscale_array.shape

    if grayscale_array.max() <= 1.0:
        grayscale_array = (grayscale_array * 255).astype(np.uint8)
    else:
        grayscale_array = grayscale_array.astype(np.uint8)

    rgb_array = np.stack([grayscale_array] * 3, axis=-1)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        f"{output_path}.mp4",
    ]

    process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    all_frames = rgb_array.tobytes()
    process.stdin.write(all_frames)
    stdout, stderr = process.communicate()


def write_yaml(data: Dict, file_name: str) -> None:
    with open(file_name, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def write_mem(data: np.ndarray, output_dir: str, filename: str = "data.mem") -> None:
    shape = data.shape
    mem_filename = os.path.join(output_dir, filename)
    mem = np.memmap(mem_filename, dtype="float64", mode="w+", shape=shape)
    mem[:] = data[:]
    mem.flush()
