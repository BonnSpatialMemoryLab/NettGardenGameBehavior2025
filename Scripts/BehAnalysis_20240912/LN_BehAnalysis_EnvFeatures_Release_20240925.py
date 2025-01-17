"""
This script analyses the behavioral information from the Garden Game task. 
For each group a text file including all information is generated. 

*Associations of environmental features with memory performance*

Laura Nett, 2024
"""

# Imports
import pandas as pd
import sys
import os
path_to_functions = '..\Functions_20250106'
sys.path.append(path_to_functions)
import LN_Functions_20240912 as LN_Functions

# Paths to get/save data
paths = {'cohort1' : 'D:\Publications\GardenGameBehavior\Data\PreProcessing\periods_complete_analysis_cohort1.csv',
         'cohort2' : 'D:\Publications\GardenGameBehavior\Data\PreProcessing\periods_complete_analysis_cohort2.csv',
         'results_cohort1' : 'D:\Publications\GardenGameBehavior\Results\Cohort1\\',
         'results_cohort2' : 'D:\Publications\GardenGameBehavior\Results\Cohort2\\',
         'results_cohort1and2' : 'D:\Publications\GardenGameBehavior\Results\Cohorts1and2\\'}

# Get dataframes 
cohort1 = pd.read_csv(paths['cohort1'])
cohort2 = pd.read_csv(paths['cohort2'])

# Remove practice trial for all analyses
cohort1 = cohort1[cohort1.TrialIdx > 0]
cohort2 = cohort2[cohort2.TrialIdx > 0]

# Combine dataframes
cohorts_1and2 = pd.concat([cohort1, cohort2])

cohort_data = [{'data': cohort1, 'path': paths['results_cohort1'], 'label': 'cohort1'},
               {'data': cohort2, 'path': paths['results_cohort2'], 'label': 'cohort2'},
               {'data': cohorts_1and2, 'path': paths['results_cohort1and2'], 'label': 'cohort1and2'}]

# Influence of environmental features
for cohort in cohort_data:
    outputfile_allo = cohort['path'] + 'allo_env_features.txt'
    outputfile_ego = cohort['path'] + 'ego_env_features.txt'
    with open(outputfile_allo, 'w') as f:
        f.write("")  # Clear the file content before running analysis
    with open(outputfile_ego, 'w') as f:
        f.write("")  # Clear the file content before running analysis
        
    # Fences
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'DistObjNearestFence', outputfile_allo)    
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'DistObjNearestFence', outputfile_ego)
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'DistObjNearestFence + DistObjPlayerStart', outputfile_ego)
    
    # North fence
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'DistObjFenceN', outputfile_allo)    
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'DistObjFenceN', outputfile_ego)
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'DistObjFenceN + DistObjPlayerStart', outputfile_ego)
    
    # Corners
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'DistObjNearestCorner', outputfile_allo)    
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'DistObjNearestCorner', outputfile_ego)
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'DistObjNearestCorner + DistObjPlayerStart', outputfile_ego)
    
    # Trees
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'DistObjNearestTree', outputfile_allo)     
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'DistObjNearestTree', outputfile_ego)

