"""
This scripts puts together period information from multiple subjects and adds several informations.

Laura Nett, 2024
"""
# Imports
import numpy as np
import pandas as pd
import sys
path_to_functions = '..\Functions_20250106'
sys.path.append(path_to_functions)
import LN_CreatePeriodsComplete_Release_20241230 as LN_CreatePeriodsComplete

# Parameters
params = {'ArenaEdgeLength' : 20, # Length of arena edge
          'CornerNE' : np.array([10, 10]), # Location of NE corner
          'CornerSE' : np.array([10, -10]), # Location of SE corner
          'CornerSW' : np.array([-10, -10]), # Location of SW corner
          'CornerNW' : np.array([-10, 10]), # Location of NW corner
          'FenceN' : np.array([np.nan, 10]), # Location of north fence
          'FenceE' : np.array([10, np.nan]), # Location of east fence
          'FenceS' : np.array([np.nan, -10]), # Location of south fence
          'FenceW' : np.array([-10, np.nan]), # Location of west fence
          'max_recall_duration' : 90000, # trials, where a recall period is longer than 90 seconds are excluded from all analyses
          'Corner_vu' : 2, # Definition of corner area for eye tracking
          'Relationship_diameter' : 1, # Diameter for gaze area
          'Fence_height' : 1.5} # Height of the fences

# Paths cohort 1
paths_cohort1 = {'logfiles'    : 'D:/Publications/GardenGameBehavior/Data/DataComplete/Cohort1/',
                 'dataframes'  : 'D:/Publications/GardenGameBehavior/Data/PreProcessing/Cohort1/',
                 'periods_complete_no_excluded' : 'D:/Publications/GardenGameBehavior/Data/PreProcessing/periods_complete_no_excluded_cohort1.csv',
                 'periods_complete_no_analysis' : 'D:/Publications/GardenGameBehavior/Data/PreProcessing/periods_complete_no_analysis_cohort1.csv',
                 'periods_complete_analysis' : 'D:/Publications/GardenGameBehavior/Data/PreProcessing/periods_complete_analysis_cohort1.csv',
                 'subjectdata' : 'SubjectData_Cohort1.csv',
                 'logfile_prefix' : 'GardenGame_',
                 'logfile_suffix' : '_LogFile.txt',
                 'periods_suffix' : '_periods.csv'}

# Paths cohort 2
paths_cohort2 = {'logfiles'    : 'D:/Publications/GardenGameBehavior/Data/DataComplete/Cohort2/',
                 'dataframes'  : 'D:/Publications/GardenGameBehavior/Data/PreProcessing/Cohort2/',
                 'periods_complete_no_excluded' : 'D:/Publications/GardenGameBehavior/Data/PreProcessing/periods_complete_no_excluded_cohort2.csv',
                 'periods_complete_no_analysis' : 'D:/Publications/GardenGameBehavior/Data/PreProcessing/periods_complete_no_analysis_cohort2.csv',
                 'periods_complete_analysis' : 'D:/Publications/GardenGameBehavior/Data/PreProcessing/periods_complete_analysis_cohort2.csv',
                 'subjectdata' : 'SubjectData_Cohort2.csv',
                 'logfile_prefix' : 'GardenGame_',
                 'logfile_suffix' : '_LogFile.txt',
                 'periods_suffix' : '_periods.csv'}

# Paths additional subjects
paths_add = {'logfiles'    : 'D:/Publications/GardenGameBehavior/Data/DataComplete/add/',
             'dataframes'  : 'D:/Publications/GardenGameBehavior/Data/PreProcessing/add/',
             'periods_complete_no_excluded' : 'D:/Publications/GardenGameBehavior/Data/PreProcessing/periods_complete_no_excluded_add.csv',
             'periods_complete_no_analysis' : 'D:/Publications/GardenGameBehavior/Data/PreProcessing/periods_complete_no_analysis_add.csv',
             'periods_complete_analysis' : 'D:/Publications/GardenGameBehavior/Data/PreProcessing/periods_complete_analysis_add.csv',
             'subjectdata' : 'SubjectData_add.csv',
             'logfile_prefix' : 'GardenGame_',
             'logfile_suffix' : '_LogFile.txt',
             'periods_suffix' : '_periods.csv'}

# Create periods complete for cohort 1
subject_data_cohort1 = pd.read_csv(paths_cohort1['logfiles'] + paths_cohort1['subjectdata'], sep = ';')
LN_CreatePeriodsComplete.get_periods_complete(paths_cohort1, params, subject_data_cohort1)

# Create periods complete for cohort 2
subject_data_cohort2 = pd.read_csv(paths_cohort2['logfiles'] + paths_cohort2['subjectdata'], sep = ';')
LN_CreatePeriodsComplete.get_periods_complete(paths_cohort2, params, subject_data_cohort2)

# Create periods complete for the additional subjects
subject_data_add = pd.read_csv(paths_add['logfiles'] + paths_add['subjectdata'], sep = ';')
LN_CreatePeriodsComplete.get_periods_complete(paths_add, params, subject_data_add)