"""
This script does all behavioral preprocessing steps for data collected from the Garden Game task, including:
- generation of dataframes (periods, timeseries, eye) per subject containing all information from the logfile
- generation of sanity check figures (Figures 2D-G and 3A) per subject
- creation of Figure 8A (if eye tracking is used) 

In addition to the log files, a subject data csv file is required, which should contain:
- SubjectName 
- SubjectIndex 
- Gender 
- Age
- EyeTracking (True or False)

Laura Nett, 2024
"""
# Imports
import sys
import os
import numpy as np
import pandas as pd
path_to_functions = '..\Functions_20250106'
sys.path.append(path_to_functions)
import LN_Functions_Release_20240912 as LN_Functions
import LN_CreateDataFramesPerSubject_Release_20240912 as LN_CreateDataFramesPerSubject
import LN_HandleEyeTrackingProblem_Release_20240912 as LN_HandleEyeTrackingProblem
import LN_Figures_Release_20241219 as LN_Figures

# Paths to get/save data
paths = {'logfiles'    : 'D:/Publications/GardenGameBehavior/Data/DataComplete/Cohort2/', # Folder with log files
         'dataframes'  : 'D:/Publications/GardenGameBehavior/Data/PreProcessing/Cohort2/', # Folder to save the dataframes
         'sanitycheckfigures' : 'D:/Publications/GardenGameBehavior/Figures/SanityCheckFigures/Cohort2/', # Folder to save the sanity check figures
         'figures' : 'D:/Publications/GardenGameBehavior/Figures/MainFigures/', # Folder to save the main figures
         'subjectdata' : 'SubjectData_Cohort2.csv', # Subject data file
         'logfile_prefix' : 'GardenGame_', # Prefix of logfiles
         'logfile_suffix' : '_LogFile.txt'} # Suffix of logfiles

# Parameters
params = {'screen_width' : 2560, # Replace with your screen width
          'screen_height' : 1600, # Replace with your screen height
          'bin_edges' : [-10,10], # bin edges for getting the percentages
          'calibration_trials' : [1,30]} # Should be changed to [1,31] if you use the Garden Game version from 2024

# Get subject data information
subject_data = pd.read_csv(paths['logfiles'] + paths['subjectdata'], sep = ';')
num_subjects = len(subject_data)

# Eye tracking dictionaries (for Figure 8A)
eye_percentages = []

# Percentage of encoding and retrieval area visited per subject
enc_percentages = {}
allo_percentages = {}
ego_percentages = {}
num_bins = 40

# Mean turning for starting position in the center vs. peripherie
center_turning = []
peripherie_turning = []

# For each subject:
for i, subject_info in subject_data.iterrows():
    # Report
    print(subject_info.SubjectName)

    # Get logfile
    logfile = paths['logfiles'] + paths['logfile_prefix'] + subject_info.SubjectName + paths['logfile_suffix']
    data = LN_Functions.get_data(logfile)
    
    # Get and save period information
    periods = LN_CreateDataFramesPerSubject.get_periods(data)
    periods.to_csv(paths['dataframes'] + str(subject_info.SubjectName) + '_periods.csv', index = False)
    
    # Get and save timeseries information
    timeseries = LN_CreateDataFramesPerSubject.get_timeseries(data)
    timeseries.to_csv(paths['dataframes'] + str(subject_info.SubjectName) + '_timeseries.csv', index = False)
    
    # If eye tracking is used, get eye tracking information
    if subject_info.EyeTracking == True:
        eye = LN_CreateDataFramesPerSubject.get_eyetracking_dataframe_continuous(logfile, periods, params['screen_width'], params['screen_height'], calibration_trials = params['calibration_trials'])
        eye.to_csv(paths['dataframes'] + str(subject_info.SubjectName) + '_eye.csv', index = False)
        
        # if there are problems with the eye tracking gaze, modify the dataframe (only needed for Garden Game version before 08/2024)
        if subject_info.EyeTrackingProblem == True:
            # if elephant is one of the animals, solve the problem
            animals = np.unique(periods[(periods.TrialIdx > 0) & (~periods.EncObj.isna())].EncObj)
            if 'Elephant' in animals:
                eye_modified = LN_HandleEyeTrackingProblem.check_and_handle_eyetracking_problems(periods, eye, 'Elephant', threshold = 0.4)
                eye_modified.to_csv(paths['dataframes'] + str(subject_info.SubjectName) + '_eye_modified.csv', index = False)
                eye = eye_modified # for sanity check figures
    
    # Create sanity check figures (2D-G & 3A)
    LN_Figures.figure2_starting_positions_and_orientations(periods, paths['sanitycheckfigures'] + str(subject_info.SubjectName) + '_2_1.svg')
    LN_Figures.figure2_enc_trajectory(periods, timeseries, paths['sanitycheckfigures'] + str(subject_info.SubjectName) + '_2_2.svg')
    LN_Figures.figure2_allo_trajectory(periods, timeseries, paths['sanitycheckfigures'] + str(subject_info.SubjectName) + '_2_3.svg')
    LN_Figures.figure2_ego_trajectory(periods, timeseries, paths['sanitycheckfigures'] + str(subject_info.SubjectName) + '_2_4.svg')
    LN_Figures.figure3A_object_locations(periods, paths['sanitycheckfigures'] + str(subject_info.SubjectName) + '_3A.svg')
    
    # Get percentages of viewed objects for figure 8A
    periods, timeseries = LN_Functions.remove_practice_trial(periods, timeseries) # Remove practice trials
 
    if subject_info.EyeTracking == True:
        # Read in file (needed because of formatting)
        if os.path.isfile(paths['dataframes'] + str(subject_info.SubjectName) + '_eye_modified.csv'):
            eye = pd.read_csv(paths['dataframes'] + str(subject_info.SubjectName) + '_eye_modified.csv')
        else:
            eye = pd.read_csv(paths['dataframes'] + str(subject_info.SubjectName) + '_eye.csv')
        perc = LN_Functions.percentages_enc_eyetracking(periods, eye)
        eye_percentages.append(perc)
        
    # Define bin edges and centers
    bins = np.linspace(params['bin_edges'][0], params['bin_edges'][1], num_bins + 1) 
    bin_centers = (bins[:-1] + bins[1:]) / 2  # midpoints of bins
    
    # Precompute valid ego bins inside the circle
    valid_ego_bins = set(
        (i, j)
        for i, x in enumerate(bin_centers)
        for j, z in enumerate(bin_centers)
        if x**2 + z**2 <= params['bin_edges'][1]**2 
    )
    
    # Bin the coordinates
    x_enc = timeseries['EncPlayerX']
    z_enc = timeseries['EncPlayerZ']
    x_allo = timeseries['AlloRetPlayerXAlloMap']
    z_allo = timeseries['AlloRetPlayerZAlloMap']
    x_ego = timeseries['EgoRetPlayerXEgoMap']
    z_ego = timeseries['EgoRetPlayerZEgoMap']
    
    x_bin_enc = np.digitize(x_enc, bins) - 1
    z_bin_enc = np.digitize(z_enc, bins) - 1
    x_bin_allo = np.digitize(x_allo, bins) - 1
    z_bin_allo = np.digitize(z_allo, bins) - 1
    x_bin_ego = np.digitize(x_ego, bins) - 1
    z_bin_ego = np.digitize(z_ego, bins) - 1
    
    # Mask out-of-bounds
    mask_enc = (x_bin_enc >= 0) & (x_bin_enc < num_bins) & (z_bin_enc >= 0) & (z_bin_enc < num_bins)
    mask_allo = (x_bin_allo >= 0) & (x_bin_allo < num_bins) & (z_bin_allo >= 0) & (z_bin_allo < num_bins)
    mask_ego = (x_bin_ego >= 0) & (x_bin_ego < num_bins) & (z_bin_ego >= 0) & (z_bin_ego < num_bins)
    
    # Occupied bins
    occupied_enc = set(zip(x_bin_enc[mask_enc], z_bin_enc[mask_enc]))
    occupied_allo = set(zip(x_bin_allo[mask_allo], z_bin_allo[mask_allo]))
    occupied_ego = set(zip(x_bin_ego[mask_ego], z_bin_ego[mask_ego]))
    
    # Valid ego bins only (within circle)
    visited_ego_bins = occupied_ego & valid_ego_bins
    
    # Compute percentages
    allo_percentages[subject_info.SubjectName] = np.round(
        len(occupied_allo) / (num_bins * num_bins) * 100, 3
    )
    ego_percentages[subject_info.SubjectName] = np.round(
        len(visited_ego_bins) / len(valid_ego_bins) * 100, 3
    )
    
    enc_percentages[subject_info.SubjectName] = np.round(
        len(occupied_enc) / (num_bins * num_bins) * 100, 3
    )
    
    # Mean turning for starting position in the center vs. peripherie
    start_pos_idx = periods[periods.PeriodType == 'starting position'].PeriodIdx.values
    filtered_timeseries = timeseries[timeseries.PeriodIdx.isin(start_pos_idx)]
    turn_peripherie, turn_center = LN_Functions.calculate_mean_turning_by_start_position(filtered_timeseries)
    center_turning.append(turn_center)
    peripherie_turning.append(turn_peripherie)

# Figure 8A (only for second cohort)
if (paths['subjectdata'] == 'SubjectData_Cohort2.csv'):
    LN_Figures.figure8A_eye_gaze_across_subjects(eye_percentages, paths['figures'] + 'Figure8A_20250606.svg')

