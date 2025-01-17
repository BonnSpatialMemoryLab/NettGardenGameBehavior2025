"""
This script analyses the behavioral information from the Garden Game task. 
For each group a text file including all information is generated.

*Age and Gender effects*

Laura Nett, 2024
"""

# Imports
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import sys
import os
path_to_functions = '..\Functions_20250106'
sys.path.append(path_to_functions)
import LN_Functions_20240912 as LN_Functions

# Paths to get/save data
paths = {'cohort1' : 'D:\Publications\GardenGameBehavior\Data\PreProcessing\periods_complete_analysis_cohort1.csv',
         'cohort2' : 'D:\Publications\GardenGameBehavior\Data\PreProcessing\periods_complete_analysis_cohort2.csv',
         'add' : 'D:\Publications\GardenGameBehavior\Data\PreProcessing\periods_complete_analysis_add.csv',
         'results_cohort1' : 'D:\Publications\GardenGameBehavior\Results\Cohort1\\',
         'results_cohort2' : 'D:\Publications\GardenGameBehavior\Results\Cohort2\\',
         'results_add' : 'D:\Publications\GardenGameBehavior\Results\Add\\',
         'results_cohort1and2' : 'D:\Publications\GardenGameBehavior\Results\Cohorts1and2\\',
         'results_complete' : 'D:\Publications\GardenGameBehavior\Results\Complete\\'}

# Get dataframes 
cohort1 = pd.read_csv(paths['cohort1'])
cohort2 = pd.read_csv(paths['cohort2'])
add = pd.read_csv(paths['add'])

# Remove practice trial for all analyses
cohort1 = cohort1[cohort1.TrialIdx > 0]
cohort2 = cohort2[cohort2.TrialIdx > 0]
add = add[add.TrialIdx > 0]

# Combine dataframes
cohorts_1and2 = pd.concat([cohort1, cohort2])
cohorts_complete = pd.concat([cohort1, cohort2, add])

cohort_data = [{'data': cohort1, 'path': paths['results_cohort1'], 'label': 'cohort1'},
               {'data': cohort2, 'path': paths['results_cohort2'], 'label': 'cohort2'},
               {'data': cohorts_1and2, 'path': paths['results_cohort1and2'], 'label': 'cohort1and2'},
               {'data': add, 'path': paths['results_add'], 'label': 'add'},
               {'data': cohorts_complete, 'path': paths['results_complete'], 'label': 'complete'}]

# Influence of age and gender
for cohort in cohort_data:
    outputfile_allo = cohort['path'] + 'allo_age_and_gender.txt'
    outputfile_ego = cohort['path'] + 'ego_age_and_gender.txt'
    with open(outputfile_allo, 'w') as f:
        f.write("")  # Clear the file content before running analysis
    with open(outputfile_ego, 'w') as f:
        f.write("")  # Clear the file content before running analysis
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'Age + Gender', outputfile_allo)
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'Age + Gender', outputfile_ego)