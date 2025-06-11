"""
This script provides a function to put together period information from multiple subjects and add several informations.

Laura Nett, 2024
"""

# Imports
import numpy as np
import pandas as pd
import os
import LN_Functions_Release_20240912 as LN_Functions

def get_periods_complete(paths, params, subject_data):
    # Add subject specific information and concatenate all period dataframes
    periods_complete = pd.DataFrame()
    for i, subject_info in subject_data.iterrows():
        # Read in periods dataframe
        periods = pd.read_csv(paths['dataframes'] + subject_info.SubjectName + paths['periods_suffix'])

        # Add subject number, age and orientation
        periods.insert(0, 'Subject', subject_info.SubjectIndex)
        periods.insert(1, 'Age', subject_info.Age)
        periods.insert(2, 'Gender', subject_info.Gender)

        # Object stability
        stable_objects = LN_Functions.get_stable_objects(periods)
        is_stable_obj = periods.EncObj.isin(stable_objects)
        mask_enc = LN_Functions.is_period_type(periods, 'encoding')
        periods.insert(12, 'StableObj', np.nan)
        periods.loc[mask_enc & (is_stable_obj), 'StableObj'] = True
        periods.loc[mask_enc & (~is_stable_obj), 'StableObj'] = False

        periods_complete = pd.concat([periods_complete, periods])
    periods_complete = periods_complete.reset_index(drop = True)

    # Add information for analyses
    is_allo_ret = LN_Functions.is_period_type(periods_complete, 'allocentric retrieval')
    is_ego_ret = LN_Functions.is_period_type(periods_complete, 'egocentric retrieval')
    is_start_pos = LN_Functions.is_period_type(periods_complete, 'starting position')
    is_enc = LN_Functions.is_period_type(periods_complete, 'encoding')

    # Change formatting of tree values 
    for column in ['TreeNW', 'TreeNE', 'TreeSW', 'TreeSE']:
        periods_complete.loc[is_enc, column] = periods_complete.loc[is_enc, column].apply(
            lambda x: np.fromstring(x.replace('[', '').replace(']', ''), dtype=float, sep=' ') if isinstance(x, str) else x)

    # Add tree configuration
    periods_complete.loc[is_enc, 'TreeConfig'] = periods_complete.loc[is_enc].apply(lambda row: ('NW_missing' if row['TreeNW'] is np.nan or row['TreeNW'] is None else
                                                                                                 'NE_missing' if row['TreeNE'] is np.nan or row['TreeNE'] is None else
                                                                                                 'SW_missing' if row['TreeSW'] is np.nan or row['TreeSW'] is None else
                                                                                                 'SE_missing'),axis=1)
    
    # Retrievals
    is_first_ego_ret = is_ego_ret & LN_Functions.is_period_type(periods_complete, 'egocentric retrieval', -1)
    is_first_allo_ret = is_allo_ret & LN_Functions.is_period_type(periods_complete, 'allocentric retrieval', -1)
    is_sec_ego_ret = is_ego_ret & LN_Functions.is_period_type(periods_complete, 'egocentric retrieval', 1)
    is_sec_allo_ret = is_allo_ret & LN_Functions.is_period_type(periods_complete, 'allocentric retrieval', 1)

    # Encoding index
    is_first_enc = is_enc & LN_Functions.is_period_type(periods_complete, 'encoding', -1)
    is_sec_enc = is_enc & LN_Functions.is_period_type(periods_complete, 'encoding', 1)
    periods_complete.loc[is_first_enc, 'EncIdx'] = 0
    periods_complete.loc[is_sec_enc, 'EncIdx'] = 1

    # Feedback 
    is_allo_first_ret_type = is_allo_ret & (LN_Functions.is_period_type(periods_complete, 'encoding', 1)|LN_Functions.is_period_type(periods_complete, 'egocentric retrieval', -1))
    is_allo_sec_ret_type = is_allo_ret & (LN_Functions.is_period_type(periods_complete, 'egocentric retrieval', 1)|LN_Functions.is_period_type(periods_complete, 'egocentric retrieval', 2))
    is_ego_first_ret_type = is_ego_ret & (LN_Functions.is_period_type(periods_complete, 'encoding', 1)|LN_Functions.is_period_type(periods_complete, 'allocentric retrieval', -1))
    is_ego_sec_ret_type = is_ego_ret & (LN_Functions.is_period_type(periods_complete, 'allocentric retrieval', 1)|LN_Functions.is_period_type(periods_complete, 'allocentric retrieval', 2))
    periods_complete.loc[(is_first_allo_ret), 'AlloWithAlloFeedback'] = False
    periods_complete.loc[(is_sec_allo_ret), 'AlloWithAlloFeedback'] = True
    periods_complete.loc[(is_first_ego_ret), 'EgoWithEgoFeedback'] = False
    periods_complete.loc[(is_sec_ego_ret), 'EgoWithEgoFeedback'] = True
    periods_complete.loc[is_allo_first_ret_type, 'AlloWithEgoFeedback'] = False
    periods_complete.loc[is_allo_sec_ret_type, 'AlloWithEgoFeedback'] = True
    periods_complete.loc[is_ego_first_ret_type, 'EgoWithAlloFeedback'] = False
    periods_complete.loc[is_ego_sec_ret_type, 'EgoWithAlloFeedback'] = True

    # Encoding object positions
    enc_obj_x = periods_complete.loc[is_enc, 'EncObjX'].values
    enc_obj_z = periods_complete.loc[is_enc, 'EncObjZ'].values

    # Starting position and orientation
    enc_player_start_x = np.repeat(periods_complete[is_start_pos].StartPosX.values, 2)
    enc_player_start_z = np.repeat(periods_complete[is_start_pos].StartPosZ.values, 2)
    enc_player_start_yaw = np.repeat(periods_complete[is_start_pos].StartPosCartYaw.values, 2)

    # Distance object to player starting position
    dist_obj_player_start = np.sqrt((enc_obj_z - enc_player_start_z)**2 + (enc_obj_x - enc_player_start_x)**2)   
    periods_complete.loc[is_enc, 'DistObjPlayerStart'] = dist_obj_player_start

    # Angle between player starting orientation and object
    vec_start_x = enc_obj_x - enc_player_start_x
    vec_start_z = enc_obj_z - enc_player_start_z
    angles_start = np.arctan2(vec_start_z, vec_start_x) - enc_player_start_yaw
    angles_start = (angles_start + np.pi) % (2*np.pi) - np.pi

    periods_complete.loc[is_enc, 'AngleObjPlayerStart'] = angles_start

    # Allocentric starting orientation
    periods_complete.loc[is_enc,'AlloStartPosOrient8Bins'] = LN_Functions.map_bins_to_orientations_allo(LN_Functions.bins_for_degrees_orientation(enc_player_start_yaw, n_bins = 8), n_bins = 8)
    periods_complete.loc[is_enc,'AlloStartPosOrient12Bins'] = LN_Functions.map_bins_to_orientations_allo(LN_Functions.bins_for_degrees_orientation(enc_player_start_yaw, n_bins = 12), n_bins = 12)

    # Egocentric orientation object to player starting position
    periods_complete.loc[is_enc,'EgoStartPosOrient'] = LN_Functions.map_bins_to_orientations_ego(LN_Functions.bins_for_degrees_orientation(angles_start, n_bins = 12), n_bins = 12)

    # Distance object to corners
    dist_obj_cornerNE = np.sqrt((params['CornerNE'][1] - enc_obj_x)**2 + (params['CornerNE'][0] - enc_obj_z)**2)
    dist_obj_cornerSE = np.sqrt((params['CornerSE'][1] - enc_obj_x)**2 + (params['CornerSE'][0] - enc_obj_z)**2)
    dist_obj_cornerSW = np.sqrt((params['CornerSW'][1] - enc_obj_x)**2 + (params['CornerSW'][0] - enc_obj_z)**2)
    dist_obj_cornerNW = np.sqrt((params['CornerNW'][1] - enc_obj_x)**2 + (params['CornerNW'][0] - enc_obj_z)**2)
    periods_complete.loc[is_enc, 'DistObjCornerNE'] = dist_obj_cornerNE
    periods_complete.loc[is_enc, 'DistObjCornerSE'] = dist_obj_cornerSE
    periods_complete.loc[is_enc, 'DistObjCornerSW'] = dist_obj_cornerSW
    periods_complete.loc[is_enc, 'DistObjCornerNW'] = dist_obj_cornerNW
    periods_complete.loc[is_enc, 'DistObjNearestCorner'] = np.min(np.stack([dist_obj_cornerNE, dist_obj_cornerSE, dist_obj_cornerSW, dist_obj_cornerNW]), axis=0)

    # Distance object to fences
    dist_obj_fenceN = np.sqrt((params['FenceN'][1] - enc_obj_z)**2)
    dist_obj_fenceE = np.sqrt((params['FenceE'][0] - enc_obj_x)**2)
    dist_obj_fenceS = np.sqrt((params['FenceS'][1] - enc_obj_z)**2)
    dist_obj_fenceW = np.sqrt((params['FenceW'][0] - enc_obj_x)**2)
    periods_complete.loc[is_enc, 'DistObjFenceN'] = dist_obj_fenceN
    periods_complete.loc[is_enc, 'DistObjFenceE'] = dist_obj_fenceE
    periods_complete.loc[is_enc, 'DistObjFenceS'] = dist_obj_fenceS
    periods_complete.loc[is_enc, 'DistObjFenceW'] = dist_obj_fenceW
    periods_complete.loc[is_enc, 'DistObjNearestFence'] = np.min(np.stack([dist_obj_fenceN, dist_obj_fenceE, dist_obj_fenceS, dist_obj_fenceW]), axis=0)

    # Distance object to trees
    treeNE_x = periods_complete.loc[is_enc, 'TreeNE'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else None)
    treeSE_x = periods_complete.loc[is_enc, 'TreeSE'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else None)
    treeSW_x = periods_complete.loc[is_enc, 'TreeSW'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else None)
    treeNW_x = periods_complete.loc[is_enc, 'TreeNW'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else None)
    treeNE_z = periods_complete.loc[is_enc, 'TreeNE'].apply(lambda x: x[2] if isinstance(x, np.ndarray) else None)
    treeSE_z = periods_complete.loc[is_enc, 'TreeSE'].apply(lambda x: x[2] if isinstance(x, np.ndarray) else None)
    treeSW_z = periods_complete.loc[is_enc, 'TreeSW'].apply(lambda x: x[2] if isinstance(x, np.ndarray) else None)
    treeNW_z = periods_complete.loc[is_enc, 'TreeNW'].apply(lambda x: x[2] if isinstance(x, np.ndarray) else None)
    dist_obj_treeNE = np.sqrt((treeNE_z - enc_obj_z)**2 + (treeNE_x - enc_obj_x)**2)
    dist_obj_treeSE = np.sqrt((treeSE_z - enc_obj_z)**2 + (treeSE_x - enc_obj_x)**2)
    dist_obj_treeSW = np.sqrt((treeSW_z - enc_obj_z)**2 + (treeSW_x - enc_obj_x)**2)
    dist_obj_treeNW = np.sqrt((treeNW_z - enc_obj_z)**2 + (treeNW_x - enc_obj_x)**2)
    periods_complete.loc[is_enc, 'DistObjTreeNE'] = dist_obj_treeNE
    periods_complete.loc[is_enc, 'DistObjTreeSE'] = dist_obj_treeSE
    periods_complete.loc[is_enc, 'DistObjTreeSW'] = dist_obj_treeSW
    periods_complete.loc[is_enc, 'DistObjTreeNW'] = dist_obj_treeNW
    periods_complete.loc[is_enc, 'DistObjNearestTree'] = np.nanmin(np.stack([dist_obj_treeNE, dist_obj_treeSE, dist_obj_treeSW, dist_obj_treeNW]), axis=0)

    # Time since end of encoding
    enc_end_times = np.repeat(periods_complete[is_sec_enc].PeriodEndTime.values, 4)
    ret_start_times = periods_complete.loc[is_allo_ret|is_ego_ret].PeriodStartTime.values
    periods_complete.loc[is_allo_ret|is_ego_ret, 'TimeSinceEncEnd'] = ret_start_times - enc_end_times
    
    # If used add eye tracking information
    if any(subject_data.EyeTracking == True):
        # Initialize all necessary columns to 0
        columns_eyetracking = ['EyeEncFence', 'EyeEncNorthFence', 'EyeEncEastFence', 'EyeEncSouthFence', 'EyeEncWestFence', 'EyeEncGazeArea', 'EyeEncTrees', 'EyeEncAnimal', 'EyeEncGazeAreaAndAnimal',
                               'EyeEncCorner', 'EyeEncNorthEastCorner', 'EyeEncSouthEastCorner', 'EyeEncSouthWestCorner', 'EyeEncNorthWestCorner', 'EyeEncCoverage']

        for col in columns_eyetracking:
            periods_complete[col] = 0
    else:
        columns_eyetracking = []
            
    # Add eye tracking for subjects with eye tracking
    for subject_name, subject_idx in zip(subject_data.SubjectName, subject_data.SubjectIndex):
        if subject_data[subject_data.SubjectIndex == subject_idx].EyeTracking.values == True:
            # Periods for this subject
            periods_subject = periods_complete[periods_complete.Subject == subject_idx]

            # Get eye tracking data for this subject
            eye_df_path = f"{paths['dataframes']}{subject_name}_eye_modified.csv"
            if os.path.exists(eye_df_path):
                # If file exist read it in
                eye_df = pd.read_csv(eye_df_path)
            else:
                eye_df = pd.read_csv(f"{paths['dataframes']}{subject_name}_eye.csv")
        else:
            print('No eye tracking data for subject ' + subject_name)
            continue

        # Add eye tracking data
        for idx, row in periods_subject.iterrows():
            if row.PeriodType == 'starting position':
                # Starting position
                player_start = np.array([row.StartPosX, row.StartPosZ])

            if row.PeriodType == 'encoding':
                # Get encoding start and stop time
                start = row.PeriodStartTime
                stop = row.PeriodEndTime

                # Get animal positions
                animal_position = np.array([row.EncObjX, row.EncObjZ])
                
                # Filter for current encoding period
                eye = eye_df[(eye_df.log_time >= start) & (eye_df.log_time <= stop)]

                if len(eye) == 0:
                    continue
                    
                # Formatting
                eye.loc[:,'gPW'] = eye.loc[:,'gPW'].apply(lambda x: np.fromstring(x.strip('[]'), sep=', '))

                # Get viewing time at specific objects
                eye_fence = eye[eye['obj'].str.contains('Fence')]
                eye_fence_north = eye_fence[eye_fence['obj'].str.contains('1.')]
                eye_fence_east = eye_fence[eye_fence['obj'].str.contains('2.')]
                eye_fence_south = eye_fence[eye_fence['obj'].str.contains('3.')]
                eye_fence_west = eye_fence[eye_fence['obj'].str.contains('4.')]
                eye_ground = eye[eye['obj'].str.contains('Ground')]
                eye_tree = eye[eye['obj'].str.contains('Tree')]
                eye_flower = eye[eye['obj'].str.contains('Flower')]
                eye_grass = eye[eye['obj'].str.contains('Grass')]
                eye_sky = eye[eye['obj'].str.contains('Sky')]

                gaze_points = np.array(eye['gPW'].tolist())
                gaze_points = gaze_points[~np.isnan(gaze_points).any(axis=1)] # only not nan gaze points
                gaze_points_ground = np.array([point[[0, 2]] for point in eye_ground.gPW])
                sum_relationship = np.sum(np.fromiter((LN_Functions.relationship_player_animal(player_start, animal_position, gaze_point, diameter = params['Relationship_diameter']) for gaze_point in gaze_points_ground), dtype=int))

                if not eye_fence.empty:
                    # Get gaze points at fences
                    gaze_points_fence = np.array([point[[0, 2]] for point in eye_fence.gPW])
                    gaze_points_fence_north = np.array([point[[0, 2]] for point in eye_fence_north.gPW]) if not eye_fence_north.empty else np.empty((0, 2))
                    gaze_points_fence_east = np.array([point[[0, 2]] for point in eye_fence_east.gPW]) if not eye_fence_east.empty else np.empty((0, 2))
                    gaze_points_fence_south = np.array([point[[0, 2]] for point in eye_fence_south.gPW]) if not eye_fence_south.empty else np.empty((0, 2))
                    gaze_points_fence_west = np.array([point[[0, 2]] for point in eye_fence_west.gPW]) if not eye_fence_west.empty else np.empty((0, 2))

                    # Add to dataframe
                    periods_complete.loc[idx, 'EyeEncFence'] = len(gaze_points_fence)/len(gaze_points)
                    periods_complete.loc[idx, 'EyeEncNorthFence'] = len(gaze_points_fence_north)/len(gaze_points) 
                    periods_complete.loc[idx, 'EyeEncEastFence'] = len(gaze_points_fence_east)/len(gaze_points) 
                    periods_complete.loc[idx, 'EyeEncSouthFence'] = len(gaze_points_fence_south)/len(gaze_points) 
                    periods_complete.loc[idx, 'EyeEncWestFence'] = len(gaze_points_fence_west)/len(gaze_points)

                # Add viewing time at gaze area
                if not eye_ground.empty:
                    periods_complete.loc[idx, 'EyeEncGazeArea'] = len(eye_tree)/len(gaze_points) 
                    
                # Add viewing time at trees
                if not eye_tree.empty:
                    periods_complete.loc[idx, 'EyeEncTrees'] = sum_relationship/len(gaze_points) 

                # Add viewing time at animal    
                obj = row.EncObj
                if len(eye[eye.obj == obj]) != 0:
                    periods_complete.loc[idx, 'EyeEncAnimal'] = len(eye[eye.obj == obj])/len(gaze_points)

                # Add viewing time at corners
                if len(gaze_points) != 0:
                    # Get gaze points at corners
                    fence_height = params['Fence_height']
                    corner_def = (params['ArenaEdgeLength']/2) - params['Corner_vu']
                    gp_corner_ne = gaze_points[(gaze_points[:, 0] >= corner_def) & (gaze_points[:, 1] <= fence_height) & (gaze_points[:, 2] >= corner_def)]
                    gp_corner_se = gaze_points[(gaze_points[:, 0] >= corner_def) & (gaze_points[:, 1] <= fence_height) & (gaze_points[:, 2] <= -corner_def)]
                    gp_corner_sw = gaze_points[(gaze_points[:, 0] <= -corner_def) & (gaze_points[:, 1] <= fence_height) & (gaze_points[:, 2] <= -corner_def)]
                    gp_corner_nw = gaze_points[(gaze_points[:, 0] <= -corner_def) & (gaze_points[:, 1] <= fence_height) & (gaze_points[:, 2] >= corner_def)]

                    # Add to dataframe
                    periods_complete.loc[idx, 'EyeEncCorner'] = (len(gp_corner_ne) + len(gp_corner_se) + len(gp_corner_sw) + len(gp_corner_nw))/len(gaze_points) 
                    periods_complete.loc[idx, 'EyeEncNorthEastCorner'] = len(gp_corner_ne)/len(gaze_points) 
                    periods_complete.loc[idx, 'EyeEncSouthEastCorner'] = len(gp_corner_se)/len(gaze_points) 
                    periods_complete.loc[idx, 'EyeEncSouthWestCorner'] = len(gp_corner_sw)/len(gaze_points)
                    periods_complete.loc[idx, 'EyeEncNorthWestCorner'] = len(gp_corner_nw)/len(gaze_points)
                
                # Add viewing time at gaze area and animal
                periods_complete.loc[idx, 'EyeEncGazeAreaAndAnimal'] = periods_complete.loc[idx, 'EyeEncAnimal'] + periods_complete.loc[idx, 'EyeEncGazeArea']
                    
        # Add eye coverage
        eye_df_filtered = LN_Functions.get_filtered_df(periods_subject, eye_df)
        indices, coverage = LN_Functions.gaze_environment_coverage(periods_subject, eye_df_filtered, n_bins = 10)
        periods_complete.loc[indices, 'EyeEncCoverage'] = coverage
    
    periods_complete.to_csv(paths['periods_complete_no_excluded'], index = False)
    
    # Identify rows where recall durations exceed the max recall duration (e.g., 90 seconds)
    exclude_df = periods_complete.loc[
        ((periods_complete['PeriodType'] == 'egocentric retrieval') |
         (periods_complete['PeriodType'] == 'allocentric retrieval')) &
        ((periods_complete['PeriodEndTime'] - periods_complete['PeriodStartTime']) > params['max_recall_duration'])
    ]

    # Create a list of (Subject, TrialIdx) combinations to exclude
    exclude = [(sub, trial) for sub, trial in zip(exclude_df['Subject'], exclude_df['TrialIdx'])]

    # Set 'AlloRetRankedPerformance' and 'EgoRetRankedPerformance' to NaN for the specified (Subject, TrialIdx) combinations
    periods_complete.loc[
        periods_complete[['Subject', 'TrialIdx']].apply(tuple, axis=1).isin(exclude),
        ['AlloRetRankedPerformance', 'EgoRetRankedPerformance']
    ] = np.nan

    # Save the modified dataframe to a new CSV
    periods_complete.to_csv(paths['periods_complete_no_analysis'], index=False)

    # Add all parameters to the corresponding retrievals for easier use in the LME
    column_names_to_fill = ['StableObj', 'EncIdx', 'TreeConfig',
                            'DistObjPlayerStart', 'AngleObjPlayerStart', 'AlloStartPosOrient8Bins','AlloStartPosOrient12Bins', 'EgoStartPosOrient',
                            'DistObjCornerNE', 'DistObjCornerSE', 'DistObjCornerSW',
                            'DistObjCornerNW', 'DistObjNearestCorner', 'DistObjFenceN',
                            'DistObjFenceE', 'DistObjFenceS', 'DistObjFenceW',
                            'DistObjNearestFence', 'DistObjTreeNE', 'DistObjTreeSE',
                            'DistObjTreeSW', 'DistObjTreeNW', 'DistObjNearestTree']
    
    for _, row in periods_complete[is_enc].iterrows():
        is_subject = (periods_complete['Subject'] == row['Subject'])
        is_trial = (periods_complete['TrialIdx'] == row['TrialIdx'])
        is_ego = (periods_complete['EgoRetObj'] == row['EncObj'])
        is_allo = (periods_complete['AlloRetObj'] == row['EncObj'])
        mask_ego = is_subject & is_trial & is_ego
        mask_allo = is_subject & is_trial & is_allo
        
        # Add time since encoding (animal specific)
        periods_complete.loc[mask_ego, 'TimeSinceEnc'] = periods_complete.loc[mask_ego, :].PeriodStartTime.values[0] - row.PeriodEndTime
        periods_complete.loc[mask_allo, 'TimeSinceEnc'] = periods_complete.loc[mask_allo, :].PeriodStartTime.values[0] - row.PeriodEndTime

        for column_name in column_names_to_fill:
            periods_complete.loc[mask_allo, column_name] = row[column_name]
            periods_complete.loc[mask_ego, column_name] = row[column_name]
        
        if len(columns_eyetracking) > 0:
            for column_name in columns_eyetracking:
                periods_complete.loc[mask_allo, column_name] = row[column_name]
                periods_complete.loc[mask_ego, column_name] = row[column_name]
            
    # Add retrieval index
    for i, row in periods_complete.iterrows():
        if row.PeriodType == 'allocentric retrieval':
            if row.AlloWithAlloFeedback and row.AlloWithEgoFeedback:
                periods_complete.loc[i, 'RetIdx'] = 4
            elif not row.AlloWithAlloFeedback and row.AlloWithEgoFeedback:
                periods_complete.loc[i, 'RetIdx'] = 3
            elif row.AlloWithAlloFeedback and not row.AlloWithEgoFeedback:
                periods_complete.loc[i, 'RetIdx'] = 2
            elif not row.AlloWithAlloFeedback and not row.AlloWithEgoFeedback:
                periods_complete.loc[i, 'RetIdx'] = 1
        elif row.PeriodType == 'egocentric retrieval':
            if row.EgoWithEgoFeedback and row.EgoWithAlloFeedback:
                periods_complete.loc[i, 'RetIdx'] = 4
            elif not row.EgoWithEgoFeedback and row.EgoWithAlloFeedback:
                periods_complete.loc[i, 'RetIdx'] = 3
            elif row.EgoWithEgoFeedback and not row.EgoWithAlloFeedback:
                periods_complete.loc[i, 'RetIdx'] = 2
            elif not row.EgoWithEgoFeedback and not row.EgoWithAlloFeedback:
                periods_complete.loc[i, 'RetIdx'] = 1
        else:
            continue
    
    # Add alignment of allocentric starting orientation with cardinal axes
    periods_complete.loc[periods_complete['AlloStartPosOrient8Bins'].isin(['N', 'E', 'S', 'W']), 'AlloStartPosAligned'] = True
    periods_complete.loc[periods_complete['AlloStartPosOrient8Bins'].isin(['NE', 'NW', 'SE', 'SW']), 'AlloStartPosAligned'] = False
    
    # Save dataframe
    periods_complete.to_csv(paths['periods_complete_analysis'], index = False)