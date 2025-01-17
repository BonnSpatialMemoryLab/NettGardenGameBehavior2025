"""
This script provides functions to create data frames (periods, timeseries, eyetracking) out of the behavioral logfile created during the Garden Game task.

Laura Nett, 2024
"""

# Imports
import LN_Functions_20240912 as LN_Functions
import numpy as np
import pandas as pd
import warnings
import math
from scipy.stats import circmean

# Function to get the period information stored in a dataframe
def get_periods(data):
    # Number of periods
    n_periods = sum(['period index' in d for d in data])

    # Column names
    column_names = ['TrialIdx', 'PeriodIdx', 'PeriodType', 'PeriodStartTime', 'PeriodEndTime', # Overall period
                    'StartPosX', 'StartPosZ', 'StartPosCartYaw', #Starting position
                    'EncObj', 'EncObjX', 'EncObjZ', # Individual encodings
                    'EncPlayerStartX', 'EncPlayerStartZ', 'EncPlayerStartUYaw', 'EncPlayerStartCartYaw', # Positions and orientations at the beginning of each encoding
                    'EgoRetObj','EgoRetObjXEgoMap', 'EgoRetObjZEgoMap', 'EgoRetObjXAlloMap', 'EgoRetObjZAlloMap', # Egocentric retrieval, object location
                    'EgoRetRespXEgoMap', 'EgoRetRespZEgoMap', 'EgoRetRespXAlloMap', 'EgoRetRespZAlloMap', # Egocentric retrieval, response location
                    'AlloRetObj','AlloRetObjXAlloMap', 'AlloRetObjZAlloMap', # Allocentric retrieval, object location
                    'AlloRetRespXAlloMap', 'AlloRetRespZAlloMap', # Allocentric retrieval, response location
                    'EgoRetDropError', 'EgoRetRankedDropError', 'EgoRetMemPoints', 'EgoRetRankedPerformance', # Egocentric performance
                    'AlloRetDropError', 'AlloRetRankedDropError', 'AlloRetRankedPerformance', 'AlloRetMemPoints', # Allocentric performance
                    'EgoRetMemAdj', 'AlloRetMemAdj', 'PlayerScore'] # Memory adjustment, total score

    # Time of all datapoints
    time = np.array([float(x.split(';')[0]) for x in data])

    # Logfile lines with "period index" are not the timepoints when the periods start but where the previous period ends
    idx_period_start_helper = [i for i, item in enumerate(data) if 'period index' in item] 
    idx_period_end_helper = idx_period_start_helper[1:] + [len(data) - 1]
    
    # Don't add the last score screen to the periods table (period index 382)
    if n_periods == 383:
        idx_period_start_helper = idx_period_start_helper[:-1]
        idx_period_end_helper = idx_period_end_helper[:-1]
        n_periods = n_periods - 1
    else: 
        # When stopping the experiment earlier, set the last valid time to the time point before the escape button is pressed
        esc_idx = [i for i, item in enumerate(data) if 'GUI; escape button pressed' in item] 
        idx_period_end_helper[-1] = int(esc_idx[0]) - 1
    
    # Change retrieval end indices to remove feedback from the retrieval period
    idx_period_end_helper_without_feedback = idx_period_end_helper.copy() 
    idx_response_given = [i for i, item in enumerate(data) if 'response given' in item] 
    idx_retrieval_end_helper = [i for i, item in enumerate(data) if 'retrieval' in item and item in [data[j] for j in idx_period_end_helper_without_feedback]]
    if len(idx_response_given) == len(idx_retrieval_end_helper):
        for idx_ret, idx_resp in zip(idx_retrieval_end_helper, idx_response_given):
            if idx_ret in idx_period_end_helper_without_feedback:
                pos = idx_period_end_helper_without_feedback.index(idx_ret) 
                idx_period_end_helper_without_feedback[pos] = idx_resp  
    else:
        raise ValueError('Error in retrieval end times')
          
    # Preallocate dataframe
    periods = pd.DataFrame(np.nan, index = range(n_periods), columns = column_names)

    # Preallocate array for the actual period start indices
    idx_period_start = np.full((len(idx_period_start_helper), 1), np.nan)

    for i_period in range(len(idx_period_start_helper)):
        # Get data from the period
        period_data = data[idx_period_start_helper[i_period]:idx_period_end_helper_without_feedback[i_period]]

        # Event markers mark where the period starts
        event_markers = {'encoding; fading in', 'encoding; memory object name', # encoding
                         'egocentric retrieval; fading in', 'egocentric retrieval; object to be retrieved', # egocentric retrieval
                         'allocentric retrieval; fading in', 'allocentric retrieval; object to be retrieved', # allocentric retrieval
                         'score; score screen started'} # score screen

        # Get period start index
        for i_period_data, item in enumerate(period_data):
            if any(marker in item for marker in event_markers):
                idx_period_start[i_period, 0] = idx_period_start_helper[i_period] + i_period_data 
                break

    # Set period start, stop time, period index and period type
    periods.PeriodStartTime = time[idx_period_start.flatten().astype(int)]
    periods.PeriodEndTime = time[np.array(idx_period_end_helper_without_feedback)]
    periods.PeriodIdx = np.array([x.split('period index: ')[1] for x in data[np.array(idx_period_start_helper)]])
    periods.PeriodType = np.array([x.split('; ')[1] for x in data[idx_period_start.flatten().astype(int)]])

    # Sanity check for the period start and end times
    if np.any((periods.PeriodEndTime - periods.PeriodStartTime) < 0):
            raise ValueError('There is a problem with the period start and end times.')

    # Get information from each period
    for i_period in range(len(periods)):
        # Data and time from this period
        period_data = data[idx_period_start_helper[i_period]:idx_period_end_helper_without_feedback[i_period]]
        period_data_with_feedback = data[idx_period_start_helper[i_period]:idx_period_end_helper[i_period]] # to get scores, response and correct positions
        period_time = np.array([x.split(';')[0] for x in period_data])

        # Get trial index
        is_trial_idx = ['trial index' in x for x in period_data]
        if sum(is_trial_idx) == 1:
            periods.loc[i_period,'TrialIdx'] = np.array([x.split('trial index: ')[1] for x in period_data[is_trial_idx]], dtype = int) # for encoding
        else:
            periods.loc[i_period,'TrialIdx'] = periods.loc[i_period - 1,'TrialIdx'] # for retrieval/score

        # Get encoding object
        is_enc_obj = ['encoding; memory object name' in x for x in period_data]
        if sum (is_enc_obj) == 1:
            periods.loc[i_period,'EncObj'] = np.array([x.split('encoding; memory object name: ')[1] for x in period_data[is_enc_obj]], dtype = str)

        # Get encoding object location
        is_enc_obj_position = ['encoding; memory object position' in x for x in period_data]
        if sum(is_enc_obj_position) == 1:
            numbers_enc_obj_position = LN_Functions.get_numbers_from_string(period_data[is_enc_obj_position.index(True)])
            periods.loc[i_period,'EncObjX'] = numbers_enc_obj_position[1]
            periods.loc[i_period,'EncObjZ'] = numbers_enc_obj_position[3]

        # Get encoding player starting position
        idx_enc_player_position = [i for i, item in enumerate(period_data) if 'encoding; player position' in item]
        if len(idx_enc_player_position) > 0:
            if (periods.PeriodType[i_period] == 'encoding') & (periods.PeriodType[i_period + 1] == 'encoding'):
                numbers_enc_player_position = LN_Functions.get_numbers_from_string(period_data[min(idx_enc_player_position)])
                periods.loc[i_period,'EncPlayerStartX'] = numbers_enc_player_position[1]
                periods.loc[i_period,'EncPlayerStartZ'] = numbers_enc_player_position[3]
                last_numbers_enc_player_position = LN_Functions.get_numbers_from_string(period_data[max(idx_enc_player_position)]) # for second encoding
            else: # exception for second encoding
                periods.loc[i_period,'EncPlayerStartX'] = last_numbers_enc_player_position[1]
                periods.loc[i_period,'EncPlayerStartZ'] = last_numbers_enc_player_position[3]

        # Get encoding player starting orientation
        idx_enc_player_uyaw = [i for i, item in enumerate(period_data) if 'encoding; player orientation' in item]
        if len(idx_enc_player_uyaw) > 0:
            if (periods.PeriodType[i_period] == 'encoding') & (periods.PeriodType[i_period + 1] == 'encoding'):
                numbers_enc_player_uyaw = LN_Functions.get_numbers_from_string(period_data[min(idx_enc_player_uyaw)])
                periods.loc[i_period,'EncPlayerStartUYaw'] = numbers_enc_player_uyaw[1]
                last_numbers_enc_player_uyaw = LN_Functions.get_numbers_from_string(period_data[max(idx_enc_player_uyaw)]) # for second encoding
            else: # exception for second encoding
                periods.loc[i_period,'EncPlayerStartUYaw'] = last_numbers_enc_player_uyaw[1]       
        elif (len(idx_enc_player_uyaw) == 0) & (periods.PeriodType[i_period] == 'encoding'):
            if math.isnan(periods.loc[i_period,'EncPlayerStartUYaw']):
                periods.loc[i_period,'EncPlayerStartUYaw'] = last_numbers_enc_player_uyaw[1]   

        # Convert to cartesian yaw radians
        periods.loc[i_period, 'EncPlayerStartCartYaw'] = LN_Functions.convert_unity_yaw_to_cart_yaw_radians(periods.loc[i_period, 'EncPlayerStartUYaw'])

        # Get egocentric retrieval object
        is_ego_ret_obj = ['egocentric retrieval; object to be retrieved' in x for x in period_data]
        if sum(is_ego_ret_obj) == 1:
            periods.loc[i_period, 'EgoRetObj'] = np.array([x.split('object to be retrieved: ')[1] for x in period_data[is_ego_ret_obj]], dtype = str)

        # Get egocentric retrieval object location on the egocentric map
        is_ego_ret_obj_ego_map = ['egocentric retrieval; correct location on the egocentric map' in x for x in period_data_with_feedback]
        if sum(is_ego_ret_obj_ego_map) == 1:
            numbers_ego_ret_obj_ego_map = LN_Functions.get_numbers_from_string(period_data_with_feedback[is_ego_ret_obj_ego_map.index(True)])
            periods.loc[i_period,'EgoRetObjXEgoMap'] = numbers_ego_ret_obj_ego_map[1]
            periods.loc[i_period,'EgoRetObjZEgoMap'] = numbers_ego_ret_obj_ego_map[3]

        # Get egocentric retrieval object location on the allocentric map
        is_ego_ret_obj_allo_map = ['egocentric retrieval; correct location on the allocentric map' in x for x in period_data_with_feedback]
        if sum(is_ego_ret_obj_allo_map) == 1:
            numbers_ego_ret_obj_allo_map = LN_Functions.get_numbers_from_string(period_data_with_feedback[is_ego_ret_obj_allo_map.index(True)])
            periods.loc[i_period,'EgoRetObjXAlloMap'] = numbers_ego_ret_obj_allo_map[1]
            periods.loc[i_period,'EgoRetObjZAlloMap'] = numbers_ego_ret_obj_allo_map[3]

        # Get egocentric retrieval response location on the egocentric map
        is_ego_ret_resp_ego_map = ['egocentric retrieval; response location on the egocentric map' in x for x in period_data_with_feedback]
        if sum(is_ego_ret_resp_ego_map) == 1:
            numbers_ego_ret_resp_ego_map = LN_Functions.get_numbers_from_string(period_data_with_feedback[is_ego_ret_resp_ego_map.index(True)])
            periods.loc[i_period,'EgoRetRespXEgoMap'] = numbers_ego_ret_resp_ego_map[1]
            periods.loc[i_period,'EgoRetRespZEgoMap'] = numbers_ego_ret_resp_ego_map[3]

        # Get egocentric retrieval response location on the allocentric map
        is_ego_ret_resp_allo_map = ['egocentric retrieval; response location on the allocentric map' in x for x in period_data_with_feedback]
        if sum(is_ego_ret_resp_allo_map) == 1:
            numbers_ego_ret_resp_allo_map = LN_Functions.get_numbers_from_string(period_data_with_feedback[is_ego_ret_resp_allo_map.index(True)])
            periods.loc[i_period,'EgoRetRespXAlloMap'] = numbers_ego_ret_resp_allo_map[1]
            periods.loc[i_period,'EgoRetRespZAlloMap'] = numbers_ego_ret_resp_allo_map[3]    

        # Get egocentric retrieval drop error
        is_ego_drop_error = ['egocentric retrieval; drop error' in x for x in period_data_with_feedback]
        if sum(is_ego_drop_error) == 1:
            periods.loc[i_period, 'EgoRetDropError'] = np.array([x.split('drop error: ')[1] for x in period_data_with_feedback[is_ego_drop_error]])

        # Get egocentric retrieval ranked drop error/performance 
        is_ego_r_drop_error = ['egocentric retrieval; ranked drop error' in x for x in period_data_with_feedback]
        if sum(is_ego_r_drop_error) == 1:
            periods.loc[i_period, 'EgoRetRankedDropError'] = np.array([x.split('ranked drop error: ')[1] for x in period_data_with_feedback[is_ego_r_drop_error]])
            periods.loc[i_period, 'EgoRetRankedPerformance'] = 1 - float(periods.loc[i_period, 'EgoRetRankedDropError'])

        # Get egocentric retrieval memory points
        is_ego_mem_points = ['egocentric retrieval; memory points' in x for x in period_data_with_feedback]
        if sum(is_ego_r_drop_error) == 1:
            periods.loc[i_period, 'EgoRetMemPoints'] = np.array([x.split('memory points: ')[1] for x in period_data_with_feedback[is_ego_mem_points]])

        # Get egocentric retrieval memory adjustment
        is_ego_mem_adj = ['egocentric retrieval; egocentric memory adjustment' in x for x in period_data_with_feedback]
        if sum(is_ego_mem_adj) > 0:
            periods.loc[i_period, 'EgoRetMemAdj'] = np.unique(np.array([x.split('egocentric memory adjustment: ')[1] for x in period_data_with_feedback[is_ego_mem_adj]]))
            
        # Get allocentric retrieval object
        is_allo_ret_obj = ['allocentric retrieval; object to be retrieved' in x for x in period_data]
        if sum(is_allo_ret_obj) == 1:
            periods.loc[i_period, 'AlloRetObj'] = np.array([x.split('object to be retrieved: ')[1] for x in period_data[is_allo_ret_obj]], dtype = str)

        # Get allocentric retrieval object location on the allocentric map
        is_allo_ret_obj_allo_map = ['allocentric retrieval; correct location' in x for x in period_data_with_feedback]
        if sum(is_allo_ret_obj_allo_map) == 1:
            numbers_allo_ret_obj_allo_map = LN_Functions.get_numbers_from_string(period_data_with_feedback[is_allo_ret_obj_allo_map.index(True)])
            periods.loc[i_period,'AlloRetObjXAlloMap'] = numbers_allo_ret_obj_allo_map[1]
            periods.loc[i_period,'AlloRetObjZAlloMap'] = numbers_allo_ret_obj_allo_map[3]

        # Get allocentric retrieval response location on the allocentric map
        is_allo_ret_resp_allo_map = ['allocentric retrieval; response location' in x for x in period_data_with_feedback]
        if sum(is_allo_ret_resp_allo_map) == 1:
            numbers_allo_ret_resp_allo_map = LN_Functions.get_numbers_from_string(period_data_with_feedback[is_allo_ret_resp_allo_map.index(True)])
            periods.loc[i_period,'AlloRetRespXAlloMap'] = numbers_allo_ret_resp_allo_map[1]
            periods.loc[i_period,'AlloRetRespZAlloMap'] = numbers_allo_ret_resp_allo_map[3]     

        # Get allocentric retrieval drop error
        is_allo_drop_error = ['allocentric retrieval; drop error' in x for x in period_data_with_feedback]
        if sum(is_allo_drop_error) == 1:
            periods.loc[i_period, 'AlloRetDropError'] = np.array([x.split('drop error: ')[1] for x in period_data_with_feedback[is_allo_drop_error]])

        # Get allocentric retrieval ranked drop error/performance 
        is_allo_r_drop_error = ['allocentric retrieval; ranked drop error' in x for x in period_data_with_feedback]
        if sum(is_allo_r_drop_error) == 1:
            periods.loc[i_period, 'AlloRetRankedDropError'] = np.array([x.split('ranked drop error: ')[1] for x in period_data_with_feedback[is_allo_r_drop_error]])
            periods.loc[i_period, 'AlloRetRankedPerformance'] = 1 - float(periods.loc[i_period, 'AlloRetRankedDropError'])

        # Get allocentric retrieval memory points
        is_allo_mem_points = ['allocentric retrieval; memory points' in x for x in period_data_with_feedback]
        if sum(is_allo_r_drop_error) == 1:
            periods.loc[i_period, 'AlloRetMemPoints'] = np.array([x.split('memory points: ')[1] for x in period_data_with_feedback[is_allo_mem_points]])

        # Get allocentric retrieval memory adjustment
        is_allo_mem_adj = ['allocentric retrieval; allocentric memory adjustment' in x for x in period_data_with_feedback]
        if sum(is_allo_mem_adj) > 0:
            periods.loc[i_period, 'AlloRetMemAdj'] = np.unique(np.array([x.split('allocentric memory adjustment: ')[1] for x in period_data_with_feedback[is_allo_mem_adj]]))

        # Get player score
        is_score = ['player score' in x for x in period_data_with_feedback]
        if sum(is_score) == 1:
            periods.loc[i_period, 'PlayerScore'] =  np.array([x.split('player score: ')[1] for x in period_data_with_feedback[is_score]])

    # Add starting position (time where starting position frame is shown)
    showing_start_pos_indices = [i for i, entry in enumerate(data) if 'GUI; showing starting position' in entry]
    not_showing_start_pos_indices = [i for i, entry in enumerate(data) if 'GUI; not showing starting position' in entry]

    # Collect lines with 'GUI; not showing starting position' and the preceding line with different time
    filtered_indices = set()
    for index in not_showing_start_pos_indices:
        current_time = data[index].split(';')[0].strip()
        for i in range(index - 1, -1, -1):
            previous_time = data[i].split(';')[0].strip()
            if previous_time != current_time:
                filtered_indices.add(i)
                break
        filtered_indices.add(index)
    filtered_indices = sorted(filtered_indices)
    filtered_data = data[filtered_indices]

    # Sanity check
    if (len(filtered_data) != 2 * len(periods.TrialIdx.unique())):
        raise ValueError('Problem in length of times!')     

    # Iterate over unique TrialIdx values
    for trial_index in periods['TrialIdx'].unique():
        trial_index_rows = periods[periods['TrialIdx'] == trial_index]

        # Get the first row for the current trial_index
        first_enc_row = trial_index_rows.iloc[0, :]

        # Create a new row for the starting position
        new_row_data = {'TrialIdx': first_enc_row['TrialIdx'],
                        'PeriodIdx': first_enc_row['PeriodIdx'],
                        'PeriodStartTime': data[showing_start_pos_indices[int(trial_index)]].split(';')[0],
                        'PeriodEndTime': filtered_data[int(2 * trial_index)].split(';')[0],
                        'PeriodType': 'starting position',
                        'StartPosX' : first_enc_row['EncPlayerStartX'],
                        'StartPosZ' : first_enc_row['EncPlayerStartZ'],
                        'StartPosCartYaw' : first_enc_row['EncPlayerStartCartYaw']}
        new_row = pd.DataFrame([new_row_data], columns = periods.columns)

        # Find the index of first encoding row and insert the new row before it
        index_to_insert = periods.index.get_loc(first_enc_row.name)

        # Change period start time of the first encoding
        periods.loc[first_enc_row.name, 'PeriodStartTime'] = filtered_data[int(2 * trial_index + 1)].split(';')[0]

        # Add new row
        periods = pd.concat([periods.iloc[:index_to_insert], new_row, periods.iloc[index_to_insert:]]).reset_index(drop=True)
    
    # Get tree information
    is_tree_info = [all(keyword in item for keyword in ['tree', 'position', 'encoding']) for item in data]
    trees = [LN_Functions.get_numbers_from_string(item) for item in np.array(data)[is_tree_info]]
    trees = [item[2:5] for item in trees]  # tree positions are at positions 2 to 5
    trees, idx = np.unique(trees, return_index = True, axis = 0)
    trees = trees[idx]
    
    # Add tree positions to encoding
    mask_enc = LN_Functions.is_period_type(periods, 'encoding')
    for tree_pos in trees:
        if (sum(np.round(tree_pos) == np.array([5,  0, 5])) == 3):
            periods.loc[mask_enc, 'TreeNE'] = periods.loc[mask_enc].apply(lambda x: tree_pos, axis=1)
        elif (sum(np.round(tree_pos) == np.array([5,  0, -5])) == 3):
            periods.loc[mask_enc, 'TreeSE'] = periods.loc[mask_enc].apply(lambda x: tree_pos, axis=1)
        elif (sum(np.round(tree_pos) == np.array([-5,  0, -5])) == 3):
            periods.loc[mask_enc, 'TreeSW'] = periods.loc[mask_enc].apply(lambda x: tree_pos, axis=1)
        elif (sum(np.round(tree_pos) == np.array([-5,  0, 5])) == 3):
            periods.loc[mask_enc, 'TreeNW'] = periods.loc[mask_enc].apply(lambda x: tree_pos, axis=1)
    return(periods)

# Function to get the timeseries information stored in a dataframe
def get_timeseries(data):
    # Time of all datapoints
    time = np.array([float(x.split(';')[0]) for x in data])
    time_hr = np.arange(min(time), max(time) + 1)
    time_hr_map = {val: idx for idx, val in enumerate(time_hr)}

    # indices of period start and end time
    idx_period_start_helper = [i for i, item in enumerate(data) if 'period index' in item]
    idx_period_end_helper = idx_period_start_helper[1:] + [len(data) - 1] 
    idx_period_start = np.full((len(idx_period_start_helper), 1), np.nan)

    for i_period in range(len(idx_period_start_helper)):
        # Get data from the period
        period_data = data[idx_period_start_helper[i_period]:idx_period_end_helper[i_period]]

        # Event markers mark where the period starts
        event_markers = {'encoding; fading in', 'encoding; memory object name', # encoding
                         'egocentric retrieval; fading in', 'egocentric retrieval; object to be retrieved', # egocentric retrieval
                         'allocentric retrieval; fading in', 'allocentric retrieval; object to be retrieved', # allocentric retrieval
                         'score; score screen started', 'score; last score screen reached'} # score screens

        # Get period start index
        for i_period_data, item in enumerate(period_data):
            if any(marker in item for marker in event_markers):
                idx_period_start[i_period, 0] = idx_period_start_helper[i_period] + i_period_data 
                break

    # Period start and stop time
    period_start_time = time[idx_period_start.flatten().astype(int)]
    period_end_time = time[np.array(idx_period_end_helper)]

    # Preallocate output table
    event_names = ['Time', 'PeriodIdx', 
                   'EncPlayerX', 'EncPlayerY', 'EncPlayerZ', 'EncPlayerUYaw', 'EncPlayerCartYaw',
                   'EgoRetPlayerXEgoMap', 'EgoRetPlayerYEgoMap', 'EgoRetPlayerZEgoMap',
                   'AlloRetPlayerXAlloMap', 'AlloRetPlayerYAlloMap', 'AlloRetPlayerZAlloMap']
    timeseries = pd.DataFrame(np.nan, index=range(len(time_hr)), columns=event_names)

    # Time
    timeseries.Time = time_hr

    # Period index
    is_this_data = ['period index' in x for x in data]
    this_data = np.array([LN_Functions.get_numbers_from_string(x) for x in data[is_this_data]])
    idx_time_hr = np.array([time_hr_map[x] for x in period_start_time])
    timeseries.loc[idx_time_hr, 'PeriodIdx'] = this_data[:,1]

    # Encoding, player position
    is_this_data = ['encoding; player position' in x for x in data]
    this_data = np.array([LN_Functions.get_numbers_from_string(x) for x in data[is_this_data]])
    idx_time_hr = np.array([time_hr_map[x] for x in this_data[:,0]])
    unique_idx_time_hr = np.unique(idx_time_hr)
    averaged_pos = np.array([np.mean(np.array(this_data[:,1:4])[idx_time_hr == t], axis=0) for t in unique_idx_time_hr])
    timeseries.loc[unique_idx_time_hr, ['EncPlayerX', 'EncPlayerY', 'EncPlayerZ']] = averaged_pos

    # Encoding, player orientation in Unity yaws and Cartesian yaws
    is_this_data = ['encoding; player orientation' in x for x in data]
    this_data = np.array([LN_Functions.get_numbers_from_string(x) for x in data[is_this_data]])
    this_data_cart_rad = LN_Functions.convert_unity_yaw_to_cart_yaw_radians(this_data[:,1])
    idx_time_hr = np.array([time_hr_map[x] for x in this_data[:,0]])
    unique_idx_time_hr = np.unique(idx_time_hr)
    averaged_cart_rad = np.array([circmean(np.array(this_data_cart_rad)[idx_time_hr == t], low = -np.pi, high = np.pi) for t in unique_idx_time_hr])
    timeseries.loc[unique_idx_time_hr, 'EncPlayerCartYaw'] = averaged_cart_rad
    timeseries.loc[unique_idx_time_hr, 'EncPlayerUYaw'] = LN_Functions.convert_cart_yaw_radians_to_unity_yaw(averaged_cart_rad)

    # Egocentric retrieval, player position
    is_this_data = ['egocentric retrieval; response position' in x for x in data]
    this_data = np.array([LN_Functions.get_numbers_from_string(x) for x in data[is_this_data]])
    idx_time_hr = np.array([time_hr_map[x] for x in this_data[:,0]])
    unique_idx_time_hr = np.unique(idx_time_hr)
    averaged_pos = np.array([np.mean(np.array(this_data[:,1:4])[idx_time_hr == t], axis=0) for t in unique_idx_time_hr])
    timeseries.loc[unique_idx_time_hr, ['EgoRetPlayerXEgoMap', 'EgoRetPlayerYEgoMap', 'EgoRetPlayerZEgoMap']] = averaged_pos

    # Allocentric retrieval, player position
    is_this_data = ['allocentric retrieval; response position' in x for x in data]
    this_data = np.array([LN_Functions.get_numbers_from_string(x) for x in data[is_this_data]])
    idx_time_hr = np.array([time_hr_map[x] for x in this_data[:,0]])
    unique_idx_time_hr = np.unique(idx_time_hr)
    averaged_pos = np.array([np.mean(np.array(this_data[:,1:4])[idx_time_hr == t], axis=0) for t in unique_idx_time_hr])
    timeseries.loc[unique_idx_time_hr, ['AlloRetPlayerXAlloMap', 'AlloRetPlayerYAlloMap', 'AlloRetPlayerZAlloMap']] = averaged_pos
    
    # Fill NaN gaps between two periods of the same type: encoding
    is_this_data = ['encoding; memory object name' in x for x in data]
    this_data = np.array([x.split(';')[0] for x in data[is_this_data]])
    last_period_idx_time = timeseries[~timeseries.PeriodIdx.isna()].Time.values
    for time_str in this_data:
        # Get last period index time and index
        time_start_fill = max((x for x in last_period_idx_time if x < float(time_str)), default=None)
        start_fill_idx = timeseries[timeseries.Time == time_start_fill].index.values[0]
        stop_fill_idx = timeseries[timeseries.Time == float(time_str)].index.values[0]

        # Forward fill 
        timeseries.loc[start_fill_idx:stop_fill_idx, 'PeriodIdx'] = timeseries.loc[start_fill_idx:stop_fill_idx, 'PeriodIdx'].ffill()
        timeseries.loc[start_fill_idx:stop_fill_idx, 'EncPlayerX'] = timeseries.loc[start_fill_idx:stop_fill_idx, 'EncPlayerX'].ffill()
        timeseries.loc[start_fill_idx:stop_fill_idx, 'EncPlayerY'] = timeseries.loc[start_fill_idx:stop_fill_idx, 'EncPlayerY'].ffill()
        timeseries.loc[start_fill_idx:stop_fill_idx, 'EncPlayerZ'] = timeseries.loc[start_fill_idx:stop_fill_idx, 'EncPlayerZ'].ffill()
        timeseries.loc[start_fill_idx:stop_fill_idx, 'EncPlayerUYaw'] = timeseries.loc[start_fill_idx:stop_fill_idx, 'EncPlayerUYaw'].ffill()
        timeseries.loc[start_fill_idx:stop_fill_idx, 'EncPlayerCartYaw'] = timeseries.loc[start_fill_idx:stop_fill_idx, 'EncPlayerCartYaw'].ffill()

    # Fill NaN gaps between two periods of the same type: allocentric
    is_this_data = ['allocentric retrieval; object to be retrieved' in x for x in data]
    this_data = np.array([x.split(';')[0] for x in data[is_this_data]])
    last_period_idx_time = timeseries[~timeseries.PeriodIdx.isna()].Time.values
    for time_str in this_data:
        # Get last period index time and index
        time_start_fill = max((x for x in last_period_idx_time if x < float(time_str)), default=None)
        start_fill_idx = timeseries[timeseries.Time == time_start_fill].index.values[0]
        stop_fill_idx = timeseries[timeseries.Time == float(time_str)].index.values[0]

        # Forward fill 
        timeseries.loc[start_fill_idx:stop_fill_idx, 'PeriodIdx'] = timeseries.loc[start_fill_idx:stop_fill_idx, 'PeriodIdx'].ffill()
        timeseries.loc[start_fill_idx:stop_fill_idx, 'EgoRetPlayerXEgoMap'] = timeseries.loc[start_fill_idx:stop_fill_idx, 'EgoRetPlayerXEgoMap'].ffill()
        timeseries.loc[start_fill_idx:stop_fill_idx, 'EgoRetPlayerYEgoMap'] = timeseries.loc[start_fill_idx:stop_fill_idx, 'EgoRetPlayerYEgoMap'].ffill()
        timeseries.loc[start_fill_idx:stop_fill_idx, 'EgoRetPlayerZEgoMap'] = timeseries.loc[start_fill_idx:stop_fill_idx, 'EgoRetPlayerZEgoMap'].ffill()

    # Fill NaN gaps between two periods of the same type: egocentric
    is_this_data = ['egocentric retrieval; object to be retrieved' in x for x in data]
    this_data = np.array([x.split(';')[0] for x in data[is_this_data]])
    last_period_idx_time = timeseries[~timeseries.PeriodIdx.isna()].Time.values
    for time_str in this_data:
        # Get last period index time and index
        time_start_fill = max((x for x in last_period_idx_time if x < float(time_str)), default=None)
        start_fill_idx = timeseries[timeseries.Time == time_start_fill].index.values[0]
        stop_fill_idx = timeseries[timeseries.Time == float(time_str)].index.values[0]

        # Forward fill 
        timeseries.loc[start_fill_idx:stop_fill_idx, 'PeriodIdx'] = timeseries.loc[start_fill_idx:stop_fill_idx, 'PeriodIdx'].ffill()
        timeseries.loc[start_fill_idx:stop_fill_idx, 'AlloRetPlayerXAlloMap'] = timeseries.loc[start_fill_idx:stop_fill_idx, 'AlloRetPlayerXAlloMap'].ffill()
        timeseries.loc[start_fill_idx:stop_fill_idx, 'AlloRetPlayerYAlloMap'] = timeseries.loc[start_fill_idx:stop_fill_idx, 'AlloRetPlayerYAlloMap'].ffill()
        timeseries.loc[start_fill_idx:stop_fill_idx, 'AlloRetPlayerZAlloMap'] = timeseries.loc[start_fill_idx:stop_fill_idx, 'AlloRetPlayerZAlloMap'].ffill()


    # Fill nan gaps''
    # Loop through periods
    for i_period in range(len(period_start_time)):
        # Data from this period
        is_this_period = (timeseries.Time >= period_start_time[i_period]) & (timeseries.Time <= period_end_time[i_period])
        period_time_series = timeseries[is_this_period]

        # Loop through variables
        for i_var in range(len(timeseries.columns)):

            # Skip if this is the time variable
            if timeseries.columns[i_var] == 'Time':
                continue

            # Data from this period
            this_period_timeseries = period_time_series.iloc[:,i_var]

            # Skip if this period only contains nans
            if this_period_timeseries.isnull().all():
                continue

            # Close initial nan gap
            idx_first_not_nan = this_period_timeseries.first_valid_index()
            this_period_timeseries.loc[:idx_first_not_nan] = this_period_timeseries[idx_first_not_nan]

            # Close nan gaps between data points
            for i_time in range(1, len(this_period_timeseries)):
                if pd.isnull(this_period_timeseries.values[i_time]):
                    this_period_timeseries.values[i_time] = this_period_timeseries.values[i_time - 1]

            # Update time series
            this_period_timeseries_start = this_period_timeseries.index[0]
            this_period_timeseries_end = this_period_timeseries.index[len(this_period_timeseries)-1] + 1
            timeseries.iloc[this_period_timeseries_start:this_period_timeseries_end, i_var] = period_time_series.iloc[:,i_var]

    return(timeseries)

# Function to get the eye tracking information stored in a dataframe
def get_eyetracking_dataframe_continuous(logfile, periods_df, screen_width, screen_height, calibration_trials):
    # Get eye tracking information out of logfile
    data = LN_Functions.get_data(logfile)
    data = data[np.where(['trial index: 1' in line for line in data])[0][0]:]
    log_eye = np.array([line for line in data if 'eye;' in line])
    split_data = [entry.split(';',3) for entry in log_eye]
    eye_df = pd.DataFrame([[sublist[0], sublist[1], sublist[3]] for sublist in split_data], columns=['log_time', 'period_information', 'information'])
    substrings_to_remove = ['hTM', 'cM', 'hide gaze', ' show gaze', 'setting up eyetracker',' bShowTrackbox', ' calibration started', ' calibration ended', 'Starting point']
    eye_df = eye_df[[not any(substring in x for substring in substrings_to_remove) for x in eye_df['information']]]
    eye_df.reset_index(inplace = True)

    # Get calibration start and end times 
    if len(calibration_trials) != 2:
        raise ValueError("There have to be exactly two calibration trials!")
    calibration1_start = int(periods_df[(periods_df.TrialIdx == calibration_trials[0]) & (periods_df.PeriodType == 'starting position')].PeriodStartTime.iloc[0])
    calibration1_stop = int(periods_df[(periods_df.TrialIdx == calibration_trials[0]) & (periods_df.PeriodType == 'starting position')].PeriodEndTime.iloc[0])
    calibration2_start = int(periods_df[(periods_df.TrialIdx == calibration_trials[1]) & (periods_df.PeriodType == 'starting position')].PeriodStartTime.iloc[0])
    calibration2_stop = int(periods_df[(periods_df.TrialIdx == calibration_trials[1]) & (periods_df.PeriodType == 'starting position')].PeriodEndTime.iloc[0])

    # Preallocate data frame
    num_unique_eye_start_times = len(eye_df[eye_df['information'].str.startswith(' lGRS')].log_time.unique())
    num_transitions = len(np.array([line for line in data if 'transition to' in line]))
    preallocated_df = pd.DataFrame(index = range(num_unique_eye_start_times + num_transitions), columns = ['log_time', 'lGRS', 'rGRS', 'gPW', 'gPS', 'obj', 'lPD', 'rPD']) 
    rows_done = 0
    c_packages = 0
    period_info = None

    # Add eye tracking information
    while rows_done < len(eye_df):
        # Check whether a new period started and fill with nans if that is the case
        if period_info != eye_df.iloc[rows_done,:].period_information:
            if rows_done != 0:
                preallocated_df.iloc[c_packages,:] = [int(log_time) + 1, [], [], [np.nan, np.nan, np.nan], [np.nan, np.nan], [np.nan], [np.nan], [np.nan]]
                c_packages += 1

        # Get log_time and period_info 
        log_time = eye_df.iloc[rows_done,:].log_time
        period_info = eye_df.iloc[rows_done,:].period_information
        
        # Identify complete packages
        b_complete_package = ('lGRS' in eye_df.iloc[rows_done,:].information) & ('rGRS' in eye_df.iloc[rows_done + 1,:].information) & ('gPW' in eye_df.iloc[rows_done + 2,:].information) & ('gPS' in eye_df.iloc[rows_done + 3,:].information) & ('obj' in eye_df.iloc[rows_done + 4,:].information) & ('lPD' in eye_df.iloc[rows_done + 5,:].information)
        if b_complete_package:
            # If log_time already in the dataframe, skip these entries
            if (c_packages > 0):
                if (preallocated_df.loc[c_packages - 1, 'log_time'] == log_time):
                    # Update index
                    rows_done += 6
                    continue

            # Extract information
            lGRS = LN_Functions.get_numbers_from_string(eye_df.iloc[rows_done,:].information, get_nans = True)
            rGRS = LN_Functions.get_numbers_from_string(eye_df.iloc[rows_done + 1,:].information, get_nans = True)
            gPW = LN_Functions.get_numbers_from_string(eye_df.iloc[rows_done + 2,:].information, get_nans = True)
            gPS = LN_Functions.get_numbers_from_string(eye_df.iloc[rows_done + 3,:].information, get_nans = True)
            obj = eye_df.iloc[rows_done + 4,:].information.split(': ')[-1]
            lPD = LN_Functions.get_numbers_from_string(eye_df.iloc[rows_done + 5,:].information, get_nans = True)[0]
            rPD = LN_Functions.get_numbers_from_string(eye_df.iloc[rows_done + 5,:].information, get_nans = True)[1]

            # Update information
            gPS = gPS[0:2] # Remove lass coordinate
            
            # Check whether gaze points are out of screen
            if (gPS[0] < 0) or (gPS[0] > (screen_width - 1)):
                gPS[0] = np.nan
            if (gPS[1] < 0) or (gPS[1] > (screen_height - 1)):
                gPS[1] = np.nan 
            if (any(np.isnan(gPS))) or (obj == 'Nothing'):
                obj = [np.nan]
            if (lPD == 0) or (np.isnan(lPD)):
                lPD = [np.nan]
            if (rPD == 0) or (np.isnan(rPD)):
                rPD = [np.nan]

            # Set calibration times to nan 
            if (calibration1_start <= int(log_time) <= calibration1_stop) | (calibration2_start <= int(log_time) <= calibration2_stop):
                preallocated_df.iloc[c_packages,:] = [log_time, [], [], [np.nan, np.nan, np.nan], [np.nan, np.nan], [np.nan], [np.nan], [np.nan]]
            
            # Set times outside of any valid period time to nan
            elif not any((periods_df['PeriodStartTime'].values.astype(float) <= float(log_time)) & (periods_df['PeriodEndTime'].values.astype(float) >= float(log_time))):
                preallocated_df.iloc[c_packages,:] = [log_time, [], [], [np.nan, np.nan, np.nan], [np.nan, np.nan], [np.nan], [np.nan], [np.nan]]
            else:
                preallocated_df.iloc[c_packages,:] = [log_time, lGRS, rGRS, gPW, [gPS[0], gPS[1]], obj, lPD, rPD]

            # Update indices
            c_packages += 1
            rows_done += 6
            
        else:
            warnings.warn('uncomplete package')
            rows_done += 1
    
    # Change type of log time
    preallocated_df.loc[:, 'log_time'] = preallocated_df['log_time'].astype('int64')

    # Make data continuous
    start = min(preallocated_df.log_time.astype('int64'))
    stop = max(preallocated_df.log_time.astype('int64'))
    continuous_log_time = pd.DataFrame({'log_time': range(start, stop + 1)})
    continuous_eyetracking = continuous_log_time.merge(preallocated_df, on='log_time', how='left')
    continuous_eyetracking = continuous_eyetracking.fillna(method='ffill')
    return(continuous_eyetracking)