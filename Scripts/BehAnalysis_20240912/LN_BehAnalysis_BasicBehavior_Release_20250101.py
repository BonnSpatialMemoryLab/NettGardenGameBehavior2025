"""
This script analyses the behavioral information from the Garden Game task. 
For each group a text file including all information is generated. 
Figures 2A-C are created with this script.

*Participant information and basic behavior*

Laura Nett, 2025
"""

# Imports
import numpy as np
import pandas as pd
import sys
import os
from scipy.stats import sem
path_to_functions = '..\Functions_20250106'
sys.path.append(path_to_functions)
import LN_Figures_20241219 as LN_Figures

# Paths to get/save data
paths = {'cohort1' : 'D:\Publications\GardenGameBehavior\Data\PreProcessing\periods_complete_no_excluded_cohort1.csv',
         'cohort2' : 'D:\Publications\GardenGameBehavior\Data\PreProcessing\periods_complete_no_excluded_cohort2.csv',
         'add' : 'D:\Publications\GardenGameBehavior\Data\PreProcessing\periods_complete_analysis_add.csv',
         'results' : 'D:\Publications\GardenGameBehavior\Results\\',
         'figures' : 'D:\Publications\GardenGameBehavior\Figures\MainFigures\\'}

# Get dataframes 
cohort1 = pd.read_csv(paths['cohort1'])
cohort2 = pd.read_csv(paths['cohort2'])
add = pd.read_csv(paths['add'])

# Combine dataframes
cohorts_1and2 = pd.concat([cohort1, cohort2])
cohorts_complete = cohorts_1and2

# Create figure 2C (trial 0 needed!)
LN_Figures.figure2C(cohorts_complete, paths['figures'] + 'Figure2C_20250116.svg')

# Remove practice trial for figures 2A and 2B
cohorts_complete = cohorts_complete[cohorts_complete.TrialIdx > 0]

# Create figures
LN_Figures.figure2A(cohorts_complete, paths['figures'] + 'Figure2A_20250116.svg')
LN_Figures.figure2B(cohorts_complete, paths['figures'] + 'Figure2B_20250116.svg')


# Participant information and basic behavior
outputfile_basic = paths['results'] + 'basic_behavior.txt'

with open(outputfile_basic, 'w') as f:   
    # Number of subjects
    f.write(f"Total number of subjects in cohort 1: {len(np.unique(cohort1.Subject))}\n")
    f.write(f"Total number of subjects in cohort 2: {len(np.unique(cohort1.Subject))}\n")
    f.write(f"Total number of subjects in additional subjects: {len(np.unique(add.Subject))}\n\n")

    # Age
    f.write("Average age in cohort 1:\n")
    f.write(f"Mean: {int(np.round(np.mean(cohort1.groupby(['Subject'])['Age'].mean())))}, ")
    f.write(f"SD: {int(np.round(np.std(cohort1.groupby(['Subject'])['Age'].mean())))}, ")
    f.write(f"Min.: {int(np.round(np.min(cohort1.groupby(['Subject'])['Age'].mean())))}, ")
    f.write(f"Max.: {int(np.round(np.max(cohort1.groupby(['Subject'])['Age'].mean())))}\n\n")
    
    f.write("Average age in cohort 2:\n")
    f.write(f"Mean: {int(np.round(np.mean(cohort2.groupby(['Subject'])['Age'].mean())))}, ")
    f.write(f"SD: {int(np.round(np.std(cohort2.groupby(['Subject'])['Age'].mean())))}, ")
    f.write(f"Min.: {int(np.round(np.min(cohort2.groupby(['Subject'])['Age'].mean())))}, ")
    f.write(f"Max.: {int(np.round(np.max(cohort2.groupby(['Subject'])['Age'].mean())))}\n\n")
    
    f.write("Average age in the additional subjects:\n")
    f.write(f"Mean: {int(np.round(np.mean(add.groupby(['Subject'])['Age'].mean())))}, ")
    f.write(f"SD: {int(np.round(np.std(add.groupby(['Subject'])['Age'].mean())))}, ")
    f.write(f"Min.: {int(np.round(np.min(add.groupby(['Subject'])['Age'].mean())))}, ")
    f.write(f"Max.: {int(np.round(np.max(add.groupby(['Subject'])['Age'].mean())))}\n\n")

    # Number of females
    f.write(f"Number of females in cohort 1: {sum(cohort1.groupby(['Subject'])['Gender'].first() == 'female')}\n")
    f.write(f"Number of females in cohort 2: {sum(cohort2.groupby(['Subject'])['Gender'].first() == 'female')}\n")
    f.write(f"Number of females in the additional subjects: {sum(add.groupby(['Subject'])['Gender'].first() == 'female')}\n\n")

    # Task duration
    task_duration_per_subject = ((cohorts_complete.groupby('Subject')['PeriodEndTime'].max() - cohorts_complete.groupby('Subject')['PeriodStartTime'].min()) / 1000 / 60).values

    f.write("Duration per session (minutes):\n")
    f.write(f"Mean: {np.round(np.mean(task_duration_per_subject),2)}, ")
    f.write(f"SEM: {np.round(sem(task_duration_per_subject),2)}, ")
    f.write(f"Min.: {np.round(np.min(task_duration_per_subject),2)}, ")
    f.write(f"Max.: {np.round(np.max(task_duration_per_subject),2)}\n\n")
    
    # Duration of periods
    cohorts_complete.loc[:, 'PeriodDuration'] = (cohorts_complete['PeriodEndTime'].astype(float) - cohorts_complete['PeriodStartTime'].astype(float)) / 1000 / 60  # in minutes

    
    # Get mean, sem, min and max for each period type across subjects
    mean_dur_per_subject = cohorts_complete.groupby(['Subject', 'PeriodType']).PeriodDuration.sum().reset_index()
    mean_dur = mean_dur_per_subject.groupby('PeriodType').PeriodDuration.mean().reset_index()
    sem_dur = mean_dur_per_subject.groupby('PeriodType').PeriodDuration.sem().reset_index()
    min_dur = mean_dur_per_subject.groupby('PeriodType').PeriodDuration.min().reset_index()
    max_dur = mean_dur_per_subject.groupby('PeriodType').PeriodDuration.max().reset_index()
    
    # Duration starting position
    f.write("Duration of starting position per session (minutes):\n")
    f.write(f"Mean: {np.round(mean_dur[mean_dur.PeriodType == 'starting position'].PeriodDuration.values[0],2)}, ")
    f.write(f"SEM: {np.round(sem_dur[sem_dur.PeriodType == 'starting position'].PeriodDuration.values[0],2)}, ")
    f.write(f"Min.: {np.round(min_dur[min_dur.PeriodType == 'starting position'].PeriodDuration.values[0],2)}, ")
    f.write(f"Max.: {np.round(max_dur[max_dur.PeriodType == 'starting position'].PeriodDuration.values[0],2)}\n\n")
    
    # Duration encoding
    f.write("Duration of encoding per session (minutes):\n")
    f.write(f"Mean: {np.round(mean_dur[mean_dur.PeriodType == 'encoding'].PeriodDuration.values[0],2)}, ")
    f.write(f"SEM: {np.round(sem_dur[sem_dur.PeriodType == 'encoding'].PeriodDuration.values[0],2)}, ")
    f.write(f"Min.: {np.round(min_dur[min_dur.PeriodType == 'encoding'].PeriodDuration.values[0],2)}, ")
    f.write(f"Max.: {np.round(max_dur[max_dur.PeriodType == 'encoding'].PeriodDuration.values[0],2)}\n\n")
    
    # Duration allocentric retrieval
    f.write("Duration of allocentric retrieval per session (minutes):\n")
    f.write(f"Mean: {np.round(mean_dur[mean_dur.PeriodType == 'allocentric retrieval'].PeriodDuration.values[0],2)}, ")
    f.write(f"SEM: {np.round(sem_dur[sem_dur.PeriodType == 'allocentric retrieval'].PeriodDuration.values[0],2)}, ")
    f.write(f"Min.: {np.round(min_dur[min_dur.PeriodType == 'allocentric retrieval'].PeriodDuration.values[0],2)}, ")
    f.write(f"Max.: {np.round(max_dur[max_dur.PeriodType == 'allocentric retrieval'].PeriodDuration.values[0],2)}\n\n")
    
    # Duration egocentric retrieval
    f.write("Duration of egocentric retrieval per session (minutes):\n")
    f.write(f"Mean: {np.round(mean_dur[mean_dur.PeriodType == 'egocentric retrieval'].PeriodDuration.values[0],2)}, ")
    f.write(f"SEM: {np.round(sem_dur[sem_dur.PeriodType == 'egocentric retrieval'].PeriodDuration.values[0],2)}, ")
    f.write(f"Min.: {np.round(min_dur[min_dur.PeriodType == 'egocentric retrieval'].PeriodDuration.values[0],2)}, ")
    f.write(f"Max.: {np.round(max_dur[max_dur.PeriodType == 'egocentric retrieval'].PeriodDuration.values[0],2)}\n\n")
    
    # Duration score screen
    f.write("Duration of encoding per session (minutes):\n")
    f.write(f"Mean: {np.round(mean_dur[mean_dur.PeriodType == 'score'].PeriodDuration.values[0],2)}, ")
    f.write(f"SEM: {np.round(sem_dur[sem_dur.PeriodType == 'score'].PeriodDuration.values[0],2)}, ")
    f.write(f"Min.: {np.round(min_dur[min_dur.PeriodType == 'score'].PeriodDuration.values[0],2)}, ")
    f.write(f"Max.: {np.round(max_dur[max_dur.PeriodType == 'score'].PeriodDuration.values[0],2)}\n\n")
    
         