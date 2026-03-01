# Allen Exporter

## Overview

This project is part of an internship at Sinzlab. The objective is to export data from the AllenSDK library into the Experanto format and use it to train existing models. This repository includes an apptainer alongside script and source code which makes the module very portable and easy to use.

## Running the Project

1. Build the apptainer image:

  ```bash
  cd allen_exporter/apptainer
  apptainer build allen_exporter.sif allen_exporter.def
  ```

2. Go to the repo root and set the environment variable:

  ```bash
  cd ../..  
  export ALLEN_BASE_DIR="$(pwd)"
  ```

3. Run the export script:

  ```bash
  ./allen-exporter/scripts/run_export.sh
  ```

  By default, this exports one experiment into a data folder next to the repo.
  To customize the export, edit allen_exporter/src/run_export.sh and check the exporter function.


## Project Structure

### Data Folder

The `data` folder which will be created during the export contains the following subdirectories:

#### 1. `brain_observatory`

This is a cache folder required to retrieve the movie data from the AllenSDK library.

#### 2. `example_experiment`

This folder contains the export of a single experiment. It is intended for testing purposes and serves as a template for the desired output structure. Details of this structure are provided below.  C

#### 3. `movies`

Contains the three different movies used in experiments. (Currently, two movies are available.)

#### 4. `visual_behaviour_cache`

This is the main cache folder that holds metadata for all experiments. It also tracks the experiments that have already been downloaded.

## Desired Structure of `example_experiment`

The `example_experiment` folder contains the following subdirectories:

### 1. `eye_tracker`

This folder stores eye-tracking data.

- **`meta.yml`**: Basic metadata about the eye-tracking experiment.
- **`data.mem`**: Contains columns with detailed measurements, including:
  - `timestamps`
  - `cr_area`, `eye_area`, `pupil_area`
  - `likely_blink`
  - `pupil_area_raw`, `cr_area_raw`, `eye_area_raw`
  - `cr_center_x`, `cr_center_y`, `cr_width`, `cr_height`, `cr_phi`
  - `eye_center_x`, `eye_center_y`, `eye_width`, `eye_height`, `eye_phi`
  - `pupil_center_x`, `pupil_center_y`, `pupil_width`, `pupil_height`, `pupil_phi`

### 2. `responses`

This folder contains neuronal response data.

- **`data/`**: Includes:
  - `cell_ids.npy`: IDs of each cell, corresponding to their rows.
  - `roi_ids.npy`: IDs of regions of interest (ROIs) to which cells belong.
  - `timestamps.npy`: Measurement times of the ophys experiment.
- **`meta.yml`**: Basic metadata about the neuronal responses.
- **`data.mem`**: Contains all activities of all measured cells.

### 3. `screen`

This folder contains data related to the stimulus presented on the screen.

- **`data/`**: Stores `.npy` files for each image/video, excluding greyscreens.
- **`meta/`**: Includes `.yml` files for each corresponding `.npy` file and for greyscreens (not saved as `.npy` files).
- **`meta.yml`**: An overarching metadata file for all screen data.

### 4. `treadmill`

This folder contains data related to the animal's running behavior.

- **`data.mem`**: Contains running speed at given timestamps.
- **`meta.yml`**: Basic metadata about the treadmill experiment.
