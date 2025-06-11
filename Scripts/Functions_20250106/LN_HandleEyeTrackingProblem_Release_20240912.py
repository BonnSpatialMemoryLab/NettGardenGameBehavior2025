"""
This script provides a function to create a modified dataframe containing the eye tracking information during the Garden Game task.

We found that there was a problem with the defined position of the elephant in Unity. This lead to wrong eye tracking results, e.g. the last hit object being the elephant with a y gaze posiition of > 2 vu even though the elephants height was smaller. We corrected for this posthoc: If the last hit object is the elephant, but the gaze is not inside the bounding box of the elephant (+ a threshold of 0.4 vu) the last hit object is set to NaN. If the last hit object is not the elephant, but the gaze is inside the bounding box, the last hit object is set to elephant. 

The available Garden Game task is already corrected for this case, so this step is not needed for new participants.

Laura Nett, 2024
"""

# Imports
import LN_Functions_Release_20240912 as LN_Functions
import numpy as np
import pandas as pd
import math

# Height, width, and length of all animals from the Garden Game task
animals = {'Bird'     : ( 1.5 *      2.54 *  0.1322,  1.5 *      2.54 *  0.6436,  1.5 *      2.54 *  0.5042), 
           'Camel'    : ( 0.8 * 0.9282798 *   2.422,  0.8 * 0.9282798 *  0.7618,  0.8 * 0.9282798 *   3.065),
           'Cat'      : ( 3.2 *      2.54 *  0.1832,  3.2 *      2.54 * 0.06259,  3.2 *      2.54 *  0.2639),
           'Chicken'  : (  5  *  9.782063 * 0.02515,    5 *  9.782063 * 0.01261,    5 *  9.782063 * 0.02557),
           'Dog'      : ( 2.2 *  16.12465 *   0.042,  2.2 *  16.12465 * 0.01438,  2.2 *  16.12465 *  0.0491),
           'Elephant' : ( 0.6 * 0.6602569 *    3.96,  0.6 * 0.6602569 *   2.326,  0.6 * 0.6602569 *   6.392),
           'Horse'    : ( 0.8 *  1.650238 *   1.192,  0.8 *  1.650238 *  0.3793,  0.8 *  1.650238 *    1.58),
           'Leopard'  : ( 1.2 *  34.30413 * 0.03391,  1.2 *  34.30413 * 0.01298,  1.2 *  34.30413 * 0.05796),
           'Penguin'  : ( 1.5 * 0.3485758 *   2.713,  1.5 * 0.3485758 *   2.388,  1.5 * 0.3485758 *   1.546),
           'Pig'      : ( 1.6 *      2.54 *  0.2878,  1.6 *      2.54 *  0.1487,  1.6 *      2.54 *  0.5247),
           'Pug'      : (  3  *      2.54 *  0.1824,    3 *      2.54 * 0.08201,    3 *      2.54 *  0.2304),
           'Rhino'    : ( 0.9 * 0.9990917 *   1.682,  0.9 * 0.9990917 *  0.8433,  0.9 * 0.9990917 *    3.22),
           'Sheep'    : (1.25 *  1.397764 *  0.8607, 1.25 *  1.397764 *  0.2917, 1.25 *  1.397764 *   1.088),
           'Tiger'    : ( 1.1 *      2.54 *  0.4948,  1.1 *      2.54 *  0.2023,  1.1 *      2.54 *  0.8213)
            }

# Get the bounding box for the object
def calculate_bounds(obj_x, obj_z, width, length):
    # Calculate orientation towards the center
    orientation = math.atan2(obj_x, -obj_z) 
    
    # Calculate the corners of the bounding box
    half_length = length / 2
    half_width = width / 2
    
    corners = [(obj_x + half_width * math.cos(orientation) - half_length * math.sin(orientation),
                obj_z + half_width * math.sin(orientation) + half_length * math.cos(orientation)),
               (obj_x + half_width * math.cos(orientation) + half_length * math.sin(orientation),
                obj_z + half_width * math.sin(orientation) - half_length * math.cos(orientation)),
               (obj_x - half_width * math.cos(orientation) + half_length * math.sin(orientation),
                obj_z - half_width * math.sin(orientation) - half_length * math.cos(orientation)),
               (obj_x - half_width * math.cos(orientation) - half_length * math.sin(orientation),
                obj_z - half_width * math.sin(orientation) + half_length * math.cos(orientation))]
    
    return corners

# Calculate whether a given point is inside a rectangle
def are_points_in_rectangle(rectangle, points, threshold = 0.4):
    # Convert rectangle vertices and points to numpy arrays 
    rectangle = np.array(rectangle)
    points = np.array(points)

    # Get the x and z coordinates of the rectangle's vertices
    x_coords = rectangle[:, 0]
    z_coords = rectangle[:, 1]

    # Calculate the min and max x and z values with threshold
    min_x = np.min(x_coords) - threshold
    max_x = np.max(x_coords) + threshold
    min_z = np.min(z_coords) - threshold
    max_z = np.max(z_coords) + threshold

    # Extract x and z coordinates of points
    px = points[:, 0]
    pz = points[:, 1]

    # Perform comparison to check if points are within bounds
    is_within_x = (min_x <= px) & (px <= max_x)
    is_within_z = (min_z <= pz) & (pz <= max_z)

    # Return array indicating which points are within the rectangle
    return is_within_x & is_within_z

# Handle problem where last hit object is not correct
def check_and_handle_eyetracking_problems(periods, eye_df, animal, threshold = 0.4):
    # Get height, width and length of animal
    animal_height = animals[animal][0]
    animal_width = animals[animal][1]
    animal_length = animals[animal][2]
    
    # Create modified eye dataframe
    eye_df_modified = eye_df.copy()
    
    # Only handle problem for encoding periods
    encoding = periods[(periods.TrialIdx > 0) & (periods.PeriodType == 'encoding')]

    for i, row in encoding.iterrows():
        # Get encoding start and end times
        period_start_time = float(row.PeriodStartTime)
        period_end_time = float(row.PeriodEndTime)

        # Get encoded animal
        encoded_animal = row.EncObj

        # Skip if encoded animal is not the animal we want to correct
        if encoded_animal != animal:
            continue

        # Get part of eye df from this period
        eye_df_period = eye_df_modified[(eye_df_modified.log_time.values.astype(float) >= period_start_time) & (eye_df_modified.log_time.values.astype(float) <= period_end_time)]

        # Get position of animal
        animal_x = row.EncObjX
        animal_z = row.EncObjZ

        # get bounds of animal
        corners = calculate_bounds(animal_x, animal_z, animal_width, animal_length)

        # Seperate eye df in the part where the object is labeled as elephant and the part where it is not
        eye_df_period_animal = eye_df_period[eye_df_period.obj == animal]
        eye_df_period_not_animal = eye_df_period[eye_df_period.obj != animal]

        # First case: labeled object is animal, but gaze points are not in the animals bounding box
        gPW_animal = eye_df_period_animal['gPW']
        gaze_animal_x = gPW_animal.apply(lambda x: x[0]).values
        gaze_animal_y = gPW_animal.apply(lambda x: x[1]).values
        gaze_animal_z = gPW_animal.apply(lambda x: x[2]).values 
        points_animal = np.column_stack((gaze_animal_x, gaze_animal_z))
        animal_wrong_case1 = np.logical_or((gaze_animal_y > animal_height), ~are_points_in_rectangle(corners, points_animal, threshold))
        animal_wrong_case1_index = eye_df_period_animal[animal_wrong_case1].index
        for index in animal_wrong_case1_index:
            eye_df_modified.loc[index, 'obj'] = [np.nan]

        # Second case: gaze points are in the animals bounding box but labeled object is not animal
        gPW_not_animal = eye_df_period_not_animal['gPW']
        gaze_not_animal_x = gPW_not_animal.apply(lambda x: x[0]).values
        gaze_not_animal_y = gPW_not_animal.apply(lambda x: x[1]).values
        gaze_not_animal_z = gPW_not_animal.apply(lambda x: x[2]).values 
        points_not_animal = np.column_stack((gaze_not_animal_x, gaze_not_animal_z))
        animal_wrong_case2 = are_points_in_rectangle(corners, points_not_animal, 0) & (gaze_not_animal_y <= animal_height)
        animal_wrong_case2_index = eye_df_period_not_animal[animal_wrong_case2].index
        eye_df_modified.loc[animal_wrong_case2_index, 'obj'] = 'Elephant'
    return(eye_df_modified)