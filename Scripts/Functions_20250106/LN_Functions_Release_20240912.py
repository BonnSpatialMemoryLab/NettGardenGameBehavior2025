"""
This script includes helping functions for the preprocessing and the analyses of the behavioral data from the Garden Game task.

Laura Nett, 2024
"""

# Imports
import numpy as np
import pandas as pd
import re
import math
import os
import statsmodels.api as sm
import pingouin as pg
from statsmodels.stats.multicomp import MultiComparison
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind
np.random.seed(42)

# Function to get data out of logfile and save it into an array
def get_data(path):
    # Read data from the text file
    with open(path, 'r') as file:
        data = file.readlines()
        
    # Strip newline characters and split each entry by semicolon
    data = [entry.strip() for entry in data]
    
    return(np.array(data))


# Functions to convert unity yaw angles to cartesian yaw radians (ranging from -pi to pi)
# because Unity has a special way of defining angles (from a bird's eye view).

# Unity:
#                     North (0°)
#                         |
#   West (-90° or 270°) -- -- East (90°)
#                         |
#                    South (180°)
#
# Cartesian:
#                     North (90°)
#                         |
#            West (180) -- -- East (0°)
#                         |
#                    South (-90°)

def convert_unity_yaw_to_cart_yaw_radians(unity_yaw):
    return(np.radians(((-1) * (unity_yaw - 90) - 180) % 360 -180))

# Function to convert cartesian yaw radians to unity yaws
def convert_cart_yaw_radians_to_unity_yaw(cart_yaw_radians):
    return((((-1 * (np.degrees(cart_yaw_radians) - 90)) % 360)))


# Function to get all numbers (and optional nans) out of a string in a list format
def get_numbers_from_string(string, get_nans = False):
    # Find numbers (and optional nans) and return as a list of floats
    if get_nans:
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?|NaN', string)
        numbers = [float(num) if num.lower() != 'nan' else np.nan for num in numbers] # Convert NaNs to actual NaN values
    else:
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?', string)
        numbers = [float(num) for num in numbers]
    return(numbers)


# Function to remove the practice trial (trial 0) from the periods and timeseries data frame
def remove_practice_trial(periods = None, timeseries = None):
    if periods is None and timeseries is None:
        raise ValueError("At least one input (periods or timeseries) must be provided.")
    
    # Process periods
    if periods is not None:
        periods = periods.loc[periods['TrialIdx'] > 0]
        periods.reset_index(drop=True, inplace=True)
    
    # Process timeseries
    if timeseries is not None:
        timeseries = timeseries.loc[timeseries['PeriodIdx'] > 6]
        timeseries.reset_index(drop=True, inplace=True)
    
    return periods, timeseries


# Function that returns the names of the stable animals
def get_stable_objects(periods):
    # Get all encoding objects
    unique_objects = np.unique(periods[~periods.EncObj.isna()].EncObj.values)

    # Preallocate array for mean distance of the encoding positions and number of occurences
    mean_distance = np.full((len(unique_objects), 1), np.nan)
    num_encodings = np.full((len(unique_objects), 1), np.nan)

    # Loop through all objects
    for i_obj, obj in enumerate(unique_objects):
        # Get all x and z positions of the object
        is_this_obj = (periods.EncObj == obj)
        obj_xz = np.column_stack((periods.loc[is_this_obj,'EncObjX'].values, periods.loc[is_this_obj,'EncObjZ'].values))

        # Get number of encoding of this object
        num_encodings[i_obj] = len(obj_xz)

        # Compute pairwise Euclidean distances between all points
        D = np.linalg.norm(obj_xz[:, np.newaxis] - obj_xz, axis = 2)

        # Only take into account upper triangle of the matrix
        mask = np.triu(np.ones(D.shape), 1) == 1
        if D[mask].size > 0:
            mean_distance[i_obj] = np.mean(D[mask])

    # Remove objects from practice trial (where animal is only encoded once)
    mask_practice_trial = num_encodings > 1
    unique_objects = unique_objects[mask_practice_trial[:, 0]]
    mean_distance = mean_distance[mask_practice_trial[:, 0]]

    # Stable objects are those where the mean distance changes minimally compared to the unstable objects
    is_stable_object = (mean_distance <= np.median(mean_distance))
    stable_objects = unique_objects[is_stable_object[:, 0]]
    return(stable_objects)

# Get bins from angles
def bins_for_degrees_orientation(angle_array, n_bins):
    # Convert nones to NaNs 
    angle_array = np.array([np.nan if x is None else x for x in angle_array])
    
    # Check for invalid degrees
    if np.any(angle_array > np.pi) or np.any(angle_array < -np.pi):
        raise ValueError("Angles must be within the range [-pi, pi]")
    
    # Define bin edges and labels
    bin_edges = np.linspace(-np.pi - np.pi/n_bins, np.pi + np.pi/n_bins, n_bins + 2)
    bin_labels = ['Bin{}'.format(i+1) for i in range(n_bins)]
    
    # Initialize the result array with NaNs
    categorized_labels = np.full(angle_array.shape, np.nan, dtype=object)
    
    # Find indices of non-NaN values
    valid_indices = ~np.isnan(angle_array)
    
    # Assign the degrees to the appropriate bins for non-NaN values
    bin_indices = (np.digitize(angle_array[valid_indices], bin_edges, right=False) - 1) % n_bins
    
    # Use the indices to get the bin labels for non-NaN values
    categorized_labels[valid_indices] = np.array(bin_labels)[bin_indices]
    
    return categorized_labels

# Map bins to allocentric orientations (N, NE, E, SE, S, SW, W, NW)
def map_bins_to_orientations_allo(binned_array):
    # Mapping of bin labels to orientations
    bin_to_orientation = {
        'Bin1': 'W',
        'Bin2': 'SW',
        'Bin3': 'S',
        'Bin4': 'SE',
        'Bin5': 'E',
        'Bin6': 'NE',
        'Bin7': 'N',
        'Bin8': 'NW'
    }
    
    # Convert bins to orientations
    orientations = [bin_to_orientation.get(bin_label, np.nan) for bin_label in binned_array]
    return(orientations)

# Map bins to egocentric orientations (A, AR, RA, R, RB, BR, B, BL, LB, L, LA, AL)
def map_bins_to_orientations_ego(binned_array):
    # Mapping of bin labels to orientations
    bin_to_orientation = {
        'Bin1': 'B',
        'Bin2': 'BR',
        'Bin3': 'RB',
        'Bin4': 'R',
        'Bin5': 'RA',
        'Bin6': 'AR',
        'Bin7': 'A',
        'Bin8': 'AL',
        'Bin9': 'LA',
        'Bin10': 'L',
        'Bin11': 'LB',
        'Bin12': 'BL'
    }
    
    # Convert bins to orientations
    orientations = [bin_to_orientation.get(bin_label, np.nan) for bin_label in binned_array]
    
    return orientations

# Check for period type
def is_period_type(periods, period_type, shift = 0):
    return((periods.PeriodType.shift(shift) == period_type))

# Linear mixed effects models
def LME(data, dependent_var, predictors, outputfile=None, mean_centered=[]):
    filtered_data = data.loc[~data[dependent_var].isna()]
    
    for i in mean_centered:
        filtered_data.loc[:,i] = filtered_data[i].values - np.mean(filtered_data[i].values)
        
    formula = f"{dependent_var} ~ {predictors}"
    model = sm.MixedLM.from_formula(formula, filtered_data, groups=filtered_data['Subject']) # Subject is the random variable
    mdf = model.fit()
    
    if outputfile is not None:
        # Check if the file exists to determine write mode
        write_mode = 'a' if os.path.exists(outputfile) else 'w'

        # Save the summary to a file
        with open(outputfile, write_mode) as f:
            # Add a header for clarity if appending
            if write_mode == 'a':
                f.write("\n\n")
            f.write(mdf.summary().as_text())
    else:
        return mdf
    
# Format results of chi-squared-test
def format_chi2_result(test_name, result):
    return f"Chi-squared test between {test_name}:\n" \
           f"  Chi2 = {result.statistic:.3f}, p = {result.pvalue:.3e}\n"

# Format results of t-test
def format_ttest_result(test_name, result):
    return f"T-test between {test_name}:\n" \
           f"  t = {result.statistic:.3f}, p = {result.pvalue:.3e}\n"

        
# Friedman test + posthoc test
def friedman_test_influence_orientation(df, performance, orientation, outputfile):
    # Filter the DataFrame to remove rows with missing performance values
    df_filtered = df.loc[~df[performance].isna(), ['Subject', 'TrialIdx', orientation, performance]]

    # Group by Subject and orientation to get mean performance per subject and orientation
    df_grouped = df_filtered.groupby(['Subject', orientation])[[performance]].mean().reset_index()

    # Get the overall mean performance for each orientation
    mean_values = df_grouped.groupby([orientation])[[performance]].mean().reset_index()

    # Friedman test to assess if performance depends on orientation
    aov = pg.friedman(dv=performance, within=orientation, subject='Subject', data=df_grouped, method='f')

     # Check if the file exists to determine write mode
    write_mode = 'a' if os.path.exists(outputfile) else 'w'

    # Open a file for writing
    with open(outputfile, write_mode) as f:
        # Write Friedman ANOVA results to file
        f.write("Friedman ANOVA Results:\n")
        f.write(aov.to_string(index=False))
        f.write("\n\n")

        # Run post-hoc comparisons only if Friedman test is significant
        if aov['p-unc'][0] < 0.05:
            f.write("Friedman test is significant. Performing post-hoc comparisons...\n\n")
            mc = MultiComparison(df_grouped[performance], df_grouped[orientation])

            # Use Sidak correction method for the post-hoc tests
            results = mc.allpairtest(stats.wilcoxon, method='s')

            # Write post-hoc test results to file
            f.write("Post-hoc Test Results:\n")
            f.write(str(results[0]))
        else:
            f.write("Friedman test is not significant. No post-hoc tests performed.\n")

            
# Feedback effects
def effect_feedback_on_performance(periods_complete):
    ego_without_allofeedback = periods_complete[periods_complete['EgoWithAlloFeedback'] == False].groupby('Subject')['EgoRetRankedPerformance'].mean()
    ego_with_allofeedback = periods_complete[periods_complete['EgoWithAlloFeedback'] == True].groupby('Subject')['EgoRetRankedPerformance'].mean()
    ego_without_egofeedback = periods_complete[periods_complete['EgoWithEgoFeedback'] == False].groupby('Subject')['EgoRetRankedPerformance'].mean()
    ego_with_egofeedback = periods_complete[periods_complete['EgoWithEgoFeedback'] == True].groupby('Subject')['EgoRetRankedPerformance'].mean()
    allo_without_egofeedback = periods_complete[periods_complete['AlloWithEgoFeedback'] == False].groupby('Subject')['AlloRetRankedPerformance'].mean()
    allo_with_egofeedback = periods_complete[periods_complete['AlloWithEgoFeedback'] == True].groupby('Subject')['AlloRetRankedPerformance'].mean()
    allo_without_allofeedback = periods_complete[periods_complete['AlloWithAlloFeedback'] == False].groupby('Subject')['AlloRetRankedPerformance'].mean()
    allo_with_allofeedback = periods_complete[periods_complete['AlloWithAlloFeedback'] == True].groupby('Subject')['AlloRetRankedPerformance'].mean()

    return(ego_without_allofeedback, ego_with_allofeedback, allo_without_egofeedback, allo_with_egofeedback,
            allo_without_allofeedback, allo_with_allofeedback, ego_without_egofeedback, ego_with_egofeedback)


# Check whether a gaze point is in the gaze area between the player's starting position and the animal's encoding location
def relationship_player_animal(player_start, animal_position, gaze_point, diameter):
    # Calculate the center of the ellipse
    center_x = (player_start[0] + animal_position[0]) / 2
    center_y = (player_start[1] + animal_position[1]) / 2
    center = (center_x, center_y)
    
    # Calculate the distance between the starting position and the animal's location
    dist = math.sqrt((player_start[0] - animal_position[0]) ** 2 + (player_start[1] - animal_position[1]) ** 2)
    
    # Calculate the angle of the line connecting the starting position and the animal's location
    angle = math.atan2(animal_position[1] - player_start[1], animal_position[0] - player_start[0])
    
    # Calculate the distance from center to the gaze point
    dx = gaze_point[0] - center_x
    dy = gaze_point[1] - center_y
    
    # Rotate the dx, dy coordinates by -angle to align with the ellipse axes
    rotated_dx = dx * math.cos(-angle) - dy * math.sin(-angle)
    rotated_dy = dx * math.sin(-angle) + dy * math.cos(-angle)
    
    # Check if the gaze point lies within the ellipse
    if (rotated_dx / (dist / 2))**2 + (rotated_dy / (diameter / 2))**2 <= 1:
        return True
    else:
        return False
    
# Create a data frame that only contains eye tracking fixation periods (continuously looking at one bin for at least 250 ms)
def get_filtered_df(periods, eye_df, threshold = 250, n_bins = 10):
    # Filter for encoding periods
    encoding = periods[periods.PeriodType == 'encoding']
    filtered_dfs = []  

    for i, row in encoding.iterrows():
        start = row.PeriodStartTime
        stop = row.PeriodEndTime

        # Filter the eye tracking dataframe for the current encoding period
        eye_df_row = eye_df[(eye_df.log_time >= start) & (eye_df.log_time <= stop)]

        # Extract gaze points
        if len(eye_df_row.gPW) == 0:
            continue

        # Only use data where the last viewed object is not NaN
        df = eye_df_row[eye_df_row.obj != '[nan]'].copy()

        # Extract x and z values
        df['x'] = df['gPW'].apply(lambda x: float(x.strip('[]').replace(',', '').split()[0]))
        df['z'] = df['gPW'].apply(lambda x: float(x.strip('[]').replace(',', '').split()[2]))

        # Define bin edges
        x_bins = np.linspace(- n_bins - 0.05, n_bins + 0.05, n_bins + 1)
        z_bins = np.linspace(- n_bins - 0.05, n_bins + 0.05, n_bins + 1)

        # Bin the x and z values
        df['x_bin'] = pd.cut(df['x'], bins=x_bins, labels=False, include_lowest=True)
        df['z_bin'] = pd.cut(df['z'], bins=z_bins, labels=False, include_lowest=True)

        # Combine x_bin and z_bin into a single bin identifier
        df['bin'] = list(zip(df['x_bin'], df['z_bin']))

        df['group'] = (df['bin'] != df['bin'].shift()).cumsum()
        group_sizes = df.groupby('group').size()
        valid_groups = group_sizes[group_sizes >= threshold].index
        filtered_df = df[df['group'].isin(valid_groups)]

        filtered_dfs.append(filtered_df)

    # Concatenate all filtered dataframes
    final_filtered_df = pd.concat(filtered_dfs, ignore_index=True)

    return final_filtered_df

# For each encoding period, get the number of bins the participant looked at
def gaze_environment_coverage(periods, eye_df, n_bins = 10):
    # Filter for encoding periods
    encoding = periods[(periods.PeriodType == 'encoding') & (periods.TrialIdx > 0)]
    
    # List to hold the number of distinct bins visited for each encoding period
    distinct_bins_visited = []
    helper_eye_df = eye_df.copy()
    for i, row in encoding.iterrows():
        start = row.PeriodStartTime
        stop = row.PeriodEndTime
        
        # Filter the eye_df for the current encoding period
        if len(helper_eye_df[helper_eye_df['log_time'] >= start]) != 0:
            eye_df_row = helper_eye_df[(helper_eye_df.log_time >= start) & (helper_eye_df.log_time <= stop)]
        else:
            distinct_bins_visited.append(0)
            continue
        
        # Get the unique bins visited during this period
        unique_bins = eye_df_row['bin'].unique()
        num_bins_visited = len(unique_bins)/(n_bins**2)
        distinct_bins_visited.append(num_bins_visited)

    return (encoding.index, distinct_bins_visited)
    
# Calculate the mean allocentric performance per subject with/without feedback
def calculate_means_allo(df):
    means = [
        df[(df.AlloWithAlloFeedback == False) & (df.AlloWithEgoFeedback == False)].groupby('Subject')['AlloRetRankedPerformance'].mean().values,
        df[(df.AlloWithAlloFeedback == True) & (df.AlloWithEgoFeedback == False)].groupby('Subject')['AlloRetRankedPerformance'].mean().values,
        df[(df.AlloWithAlloFeedback == False) & (df.AlloWithEgoFeedback == True)].groupby('Subject')['AlloRetRankedPerformance'].mean().values,
        df[(df.AlloWithAlloFeedback == True) & (df.AlloWithEgoFeedback == True)].groupby('Subject')['AlloRetRankedPerformance'].mean().values
    ]
    return means

# Calculate the mean egocentric performance per subject with/without feedback
def calculate_means_ego(df):
    means = [
        df[(df.EgoWithEgoFeedback == False) & (df.EgoWithAlloFeedback == False)].groupby('Subject')['EgoRetRankedPerformance'].mean().values,
        df[(df.EgoWithEgoFeedback == True) & (df.EgoWithAlloFeedback == False)].groupby('Subject')['EgoRetRankedPerformance'].mean().values,
        df[(df.EgoWithEgoFeedback == False) & (df.EgoWithAlloFeedback == True)].groupby('Subject')['EgoRetRankedPerformance'].mean().values,
        df[(df.EgoWithEgoFeedback == True) & (df.EgoWithAlloFeedback == True)].groupby('Subject')['EgoRetRankedPerformance'].mean().values
    ]
    return means


# Define the Weibull function
def weibull(x, k, lambd, c):
    return c * (1 - np.exp(-(x / lambd) * k)) 

# Function to fit Weibull function
def fit_weibull(x, y):
    # Initial guess for Weibull parameters
    initial_guess = [2.0, 10.0, 1.0]  
    try:
        # Fit curve
        popt, _ = curve_fit(weibull, x, y, p0=initial_guess, maxfev=10000, bounds = ([0.01, 0.01, 0], [100, 20, 1]))
        y_pred = weibull(x, *popt)

        return popt, y_pred
    except RuntimeError:
        # Return None if fitting fails
        return None, None

# Create table with weibull parameters
def create_weibull_table(periods_complete):
    # Preallocate dataframe
    columns = ['subject_id', 'animal', 'ret_type', 'object_stability', 'k', 'lambda', 'c']
    weibull_table = pd.DataFrame(index = np.arange(2 * 6 * 64), columns = columns)

    # Get subject-object combinations
    enc_periods = periods_complete[periods_complete.PeriodType == 'encoding']
    sub_obj_comb = [f"{row['Subject']}_{row['EncObj']}" for _, row in enc_periods.iterrows()]
    sub_obj_comb = np.unique(np.array(sub_obj_comb))

    # Counter for weibull table
    n_count = 0
    
    # Loop through all subject-object combinations
    for combo in sub_obj_comb:
        # Split the combination back into Subject and Object
        subject, obj = combo.split('_')

        # Get data
        subject_data_allo = periods_complete[(periods_complete.Subject == int(subject)) & (periods_complete.AlloRetObj == obj)]
        subject_data_ego = periods_complete[(periods_complete.Subject == int(subject)) & (periods_complete.EgoRetObj == obj)]
        
        # Remove excluded trials (where performance is nan)
        subject_data_allo = subject_data_allo[~subject_data_allo.AlloRetRankedPerformance.isna()]
        subject_data_ego = subject_data_ego[~subject_data_ego.EgoRetRankedPerformance.isna()]

        # Number of trials
        n_trials = len(subject_data_allo)
        
        # Trial numbers
        x = np.arange(1, n_trials + 1) 

        # Allocentric
        y_allo = subject_data_allo.AlloRetRankedPerformance.values
        empirical_params_allo, empirical_fit_allo = fit_weibull(x, y_allo)

        # Egocentric
        y_ego = subject_data_ego.EgoRetRankedPerformance.values
        empirical_params_ego, empirical_fit_ego = fit_weibull(x, y_ego)

        # Add values for allocentric learning curves to weibull table
        weibull_table.loc[n_count, 'subject_id'] = subject
        weibull_table.loc[n_count, 'animal'] = obj
        weibull_table.loc[n_count, 'ret_type'] = 'allo'
        weibull_table.loc[n_count, 'object_stability'] = np.unique(subject_data_allo.StableObj)[0]
        weibull_table.loc[n_count, 'k'] = empirical_params_allo[0]
        weibull_table.loc[n_count, 'lambda'] = empirical_params_allo[1]
        weibull_table.loc[n_count, 'c'] = empirical_params_allo[2]

        # Add values for egocentric learning curves to weibull table
        weibull_table.loc[n_count + 1, 'subject_id'] = subject
        weibull_table.loc[n_count + 1, 'animal'] = obj
        weibull_table.loc[n_count + 1, 'ret_type'] = 'ego'
        weibull_table.loc[n_count + 1, 'object_stability'] = np.unique(subject_data_allo.StableObj)[0]
        weibull_table.loc[n_count + 1, 'k'] = empirical_params_ego[0]
        weibull_table.loc[n_count + 1, 'lambda'] = empirical_params_ego[1]
        weibull_table.loc[n_count + 1, 'c'] = empirical_params_ego[2]
        n_count += 2
    return(weibull_table)

# Add trials of change points, slope and initial performance
def get_change_points(periods_complete, weibull_table):
    # Get data for this subject and animal
    for i, row in weibull_table.iterrows():
        # Get information from weibull table
        subject = int(row.subject_id)
        animal = row.animal
        ret_type = row.ret_type
        k_emp = row.k
        lambda_emp = row['lambda']
        c_emp = row.c

        # Get performance values
        if ret_type == 'allo':
            data = periods_complete[(periods_complete.Subject == subject) & (periods_complete.AlloRetObj == animal)]
            data = data[~data.AlloRetRankedPerformance.isna()]
            y_emp = data.AlloRetRankedPerformance.values
        elif ret_type == 'ego':
            data = periods_complete[(periods_complete.Subject == subject) & (periods_complete.EgoRetObj == animal)]
            data = data[~data.EgoRetRankedPerformance.isna()]
            y_emp = data.EgoRetRankedPerformance.values
            
        # Get values of fitted weibull curve
        x_values = np.arange(1, len(data) + 1) # 1-20/1-19
        y_weibull_emp = weibull(x_values, k_emp, lambda_emp, c_emp)

        # Add initial performance to weibull table
        weibull_table.loc[i,'initial_performance'] = y_weibull_emp[0]

        # If the variance in the data is to small, skip the t-tests (no change point detected)
        if np.var(y_weibull_emp) < 1e-3:
            weibull_table.loc[i,'change_point'] = np.nan
            weibull_table.loc[i,'is_change_point'] = False
            weibull_table.loc[i,'slope'] = 0
            continue

        # Initialize lists to store t-statistics 
        t_statistics_emp = []
        t_statistics_surr = []

        # Perform t-tests on empirical data for all possible trial split points
        for trial_idx in x_values - 1: # 0-19/0-18
            # Split data in first and second half
            first_half = y_weibull_emp[:trial_idx + 1]
            sec_half = y_weibull_emp[trial_idx + 1:]

            # Perform t-test
            t_stat, p_value = ttest_ind(sec_half, first_half)
            t_statistics_emp.append(t_stat)

        # Perform t-test on surrogate data
        num_surr = 1000
        for i_surr in np.arange(0,num_surr):
            # Shuffle data
            y_shuffled = np.random.permutation(y_emp)

            # Fit weibull function
            surr_params, surr_fit = fit_weibull(x_values, y_shuffled)
            surr_k = surr_params[0]
            surr_lambda = surr_params[1]
            surr_c = surr_params[2]
            y_weibull_surr = weibull(x_values, surr_k, surr_lambda, surr_c)

            # If the variance in the data is to small, skip the t-tests 
            if np.var(y_weibull_surr) < 1e-3:
                t_statistics_surr.append(np.array(len(x_values) * [np.nan]))
                continue

            # t-statistics for this surrogate round
            t_statistics_this_surr = []
            for trial_idx in x_values - 1: # 0-19/0-18
                # Split data in first and second half
                first_half = y_weibull_surr[:trial_idx + 1]
                sec_half = y_weibull_surr[trial_idx + 1:]

                # Perform t-test
                t_stat, p_value = ttest_ind(sec_half, first_half)
                t_statistics_this_surr.append(t_stat)

            # Append the t-statistics
            t_statistics_surr.append(t_statistics_this_surr)

        # Transfer lists to arrays
        t_statistics_emp = np.array(t_statistics_emp)
        t_statistics_surr = np.array(t_statistics_surr)
        t_statistics_surr = t_statistics_surr.T

        # Get change point based on empirical t-statistic versus surrogate t-statistics
        ranks = []
        for j in np.arange(len(t_statistics_surr)):
            # Get t-statistics for one trial
            t_stats_surr = t_statistics_surr[j]

            # Empirical t-value 
            tstat_emp = t_statistics_emp[j]

            # Remove nans and infinite values
            clean_arr = t_stats_surr[np.isfinite(t_stats_surr)]

            # Calculate rank (proportion of surrogate values being smaller than the observed value)
            if len(clean_arr) != 0:
                rank = np.sum(tstat_emp > clean_arr)/len(clean_arr) 
            else:
                rank = np.nan
            ranks.append(rank)

        # Get index where rank is highest
        change_point = np.nanargmax(ranks) + 1 # trial
        weibull_table.loc[i,'change_point'] = change_point
        weibull_table.loc[i,'is_change_point'] = True

        # Slope at change point
        dy_dx = np.gradient(y_weibull_emp, x_values)  # Compute the first derivative (slope) 
        slope = dy_dx[change_point - 1]
        weibull_table.loc[i,'slope'] = slope
    return(weibull_table)


# Viewed objects during encoding
def percentages_enc_eyetracking(periods, eye):
    # Remove practice trial
    periods, _ = remove_practice_trial(periods, None)
    
    # Get eye tracking information from the encoding
    encoding = periods[periods.PeriodType == 'encoding']
    encoding_times = np.array([])
    for i, row in encoding.iterrows():
        period_times = np.arange(float(row.PeriodStartTime), float(row.PeriodEndTime) + 1)
        encoding_times = np.concatenate((encoding_times, period_times))
    eye_enc = eye[eye['log_time'].isin(encoding_times)]
    
    # Get gaze objects
    gaze_object_encoding = eye_enc.obj.values
    
    # Get animals 
    animals = np.unique(periods[~periods.EncObj.isna()].EncObj)
    
    # Define corresponding lables
    encoding_objects = {'Objects' : [animals[0], animals[1], animals[2], animals[3], animals[4], animals[5]], 
                        'North fence' : ['Fence1.0', 'FenceHelper1.0', 'FenceHelper1.1', 'FenceHelper1.2','FenceHelper1.3', 'FenceHelper1.4', 'FenceHelper1.5', 'FenceHelper1.6',
                                         'intermittentFences1.1', 'intermittentFences1.2', 'intermittentFences1.3', 'intermittentFences1.4', 'intermittentFences1.5', 
                                         'intermittentFences1.6'],  
                        'East fence' :  ['Fence2.0', 'FenceHelper2.0', 'FenceHelper2.1', 'FenceHelper2.2','FenceHelper2.3', 'FenceHelper2.4', 'FenceHelper2.5', 'FenceHelper2.6',
                                         'intermittentFences2.1', 'intermittentFences2.2', 'intermittentFences2.3', 'intermittentFences2.4', 'intermittentFences2.5', 
                                         'intermittentFences2.6'],  
                        'South fence' : ['Fence3.0', 'FenceHelper3.0', 'FenceHelper3.1', 'FenceHelper3.2','FenceHelper3.3', 'FenceHelper3.4', 'FenceHelper3.5', 'FenceHelper3.6',
                                         'intermittentFences3.1', 'intermittentFences3.2', 'intermittentFences3.3', 'intermittentFences3.4', 'intermittentFences3.5', 
                                         'intermittentFences3.6'],   
                        'West fence' :  ['Fence4.0', 'FenceHelper4.0', 'FenceHelper4.1', 'FenceHelper4.2','FenceHelper4.3', 'FenceHelper4.4', 'FenceHelper4.5', 'FenceHelper4.6',
                                         'intermittentFences4.1', 'intermittentFences4.2', 'intermittentFences4.3', 'intermittentFences4.4', 'intermittentFences4.5', 
                                         'intermittentFences4.6'],
                        'Flowers' : ['Flower0.1', 'Flower0.2', 'Flower1.1', 'Flower1.2', 'Flower2.1', 'Flower2.2', 
                                     'ThreeFlowers0.1', 'ThreeFlowers0.2', 'ThreeFlowers0.3', 'ThreeFlowers0.4', 
                                     'ThreeFlowers1.1', 'ThreeFlowers1.2', 'ThreeFlowers1.3', 'ThreeFlowers1.4', 
                                     'ThreeFlowers2.1', 'ThreeFlowers2.2', 'ThreeFlowers2.3', 'ThreeFlowers2.4', 
                                     'ThreeFlower0.1', 'ThreeFlower0.2', 'ThreeFlower0.3', 'ThreeFlower0.4', 
                                     'ThreeFlower1.1', 'ThreeFlower1.2', 'ThreeFlower1.3', 'ThreeFlower1.4', 
                                     'ThreeFlower2.1', 'ThreeFlower2.2', 'ThreeFlower2.3', 'ThreeFlower2.4'],
                        'Ground' : ['Ground', 'GroundTerrain'],
                        'Trees' : ['Tree0', 'Tree1', 'Tree2'],   
                        'Sky' : ['Sky','SkyHelper1', 'SkyHelper2', 'SkyHelper3', 'SkyHelper4'],   
                        'Grass' : ['LittleGrass0.0', 'LittleGrass0.1', 'LittleGrass0.2', 'LittleGrass1.0', 'LittleGrass1.1', 'LittleGrass1.2', 'LittleGrass2.0', 'LittleGrass2.1',
                                   'LittleGrass2.2'], 
                        'Handles' : ['Handles', 'hand_left', 'hand_right']}
    
    # Get corresponding labels for gaze objects
    object_list = []
    not_found_list = []
    for gaze_object in gaze_object_encoding:
        if gaze_object != '[nan]':
            # Iterate through the dictionary to find the key
            for key, values_list in encoding_objects.items():
                if gaze_object in values_list:
                    object_list.append(key)
                    break  
            else:
                print(gaze_object.dtype)
                not_found_list.append(gaze_object)
    object_array = np.array(object_list)
    
    # Sanity check
    if len(not_found_list) > 0:
        print(not_found_list)
        unique_not_found = set(not_found_list)
        print("Unique values not found in encoding_objects:", unique_not_found)
        raise ValueError("Some gaze objects were not found in the encoding_objects dictionary.")
    
    # Calculate percentages
    unique, counts = np.unique(object_array, return_counts=True)
    object_counts = dict(zip(unique, counts))
    total_count = len(object_array)
    percentages = {obj: (count / total_count) * 100 for obj, count in object_counts.items()}
    return(percentages)

# Helping function for Figure 8A
def collect_values_by_key(dict_list):
    # Create a set of all unique keys across all dictionaries
    all_keys = set().union(*dict_list)
    
    # Collect all values for each key in a list
    dict_together = {key: [] for key in all_keys}
    
    for d in dict_list:
        for key in all_keys:
            # Append each value, defaulting to 0 if missing
            dict_together[key].append(d.get(key, 0))  
    
    return dict_together

