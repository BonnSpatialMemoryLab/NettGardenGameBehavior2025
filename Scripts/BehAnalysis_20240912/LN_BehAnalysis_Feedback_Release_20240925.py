"""
This script analyses the behavioral information from the Garden Game task. 
For each group a text file including all information is generated. 
Figures 7B-D are created with this script.

*Associations between feedback and memory performance*

Laura Nett, 2024 
"""

# Imports
import pandas as pd
import sys
import os
path_to_functions = '..\Functions_20250106'
sys.path.append(path_to_functions)
import LN_Functions_20240912 as LN_Functions
import LN_Figures_20241219 as LN_Figures

# Paths to get/save data
paths = {'cohort1' : 'D:\Publications\GardenGameBehavior\Data\PreProcessing\periods_complete_analysis_cohort1.csv',
         'cohort2' : 'D:\Publications\GardenGameBehavior\Data\PreProcessing\periods_complete_analysis_cohort2.csv',
         'results_cohort1' : 'D:\Publications\GardenGameBehavior\Results\Cohort1\\',
         'results_cohort2' : 'D:\Publications\GardenGameBehavior\Results\Cohort2\\',
         'results_cohort1and2' : 'D:\Publications\GardenGameBehavior\Results\Cohorts1and2\\',
         'figures' : 'D:\Publications\GardenGameBehavior\Figures\MainFigures\\'}

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

# Influence of feedback and time since encoding
for cohort in cohort_data:
    outputfile_allo = cohort['path'] + 'allo_feedback.txt'
    outputfile_ego = cohort['path'] + 'ego_feedback.txt'
    with open(outputfile_allo, 'w') as f:
        f.write("")  # Clear the file content before running analysis
    with open(outputfile_ego, 'w') as f:
        f.write("")  # Clear the file content before running analysis
        
    # Influence of feedback
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'AlloWithAlloFeedback', cohort['path'] + 'allo_feedback.txt')
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'AlloWithEgoFeedback', cohort['path'] + 'allo_feedback.txt')
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'EgoWithEgoFeedback', cohort['path'] + 'ego_feedback.txt')
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'EgoWithAlloFeedback', cohort['path'] + 'ego_feedback.txt')
    
    # Influence of retrieval index 
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'C(RetIdx)', cohort['path'] + 'allo_feedback.txt')
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'C(RetIdx)', cohort['path'] + 'ego_feedback.txt')
    
    # Influence of time elapsed since end of encoding 
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'TimeSinceEncEnd', cohort['path'] + 'allo_feedback.txt')
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'TimeSinceEncEnd', cohort['path'] + 'ego_feedback.txt')
    
    # Influence of time elapsed since encoding of specific animal
    LN_Functions.LME(cohort['data'][cohort['data'].EncIdx == 0], 'AlloRetRankedPerformance', 'TimeSinceEnc', cohort['path'] + 'allo_feedback.txt')
    LN_Functions.LME(cohort['data'][cohort['data'].EncIdx == 1], 'AlloRetRankedPerformance', 'TimeSinceEnc', cohort['path'] + 'allo_feedback.txt')
    LN_Functions.LME(cohort['data'][cohort['data'].EncIdx == 0], 'EgoRetRankedPerformance', 'TimeSinceEnc', cohort['path'] + 'ego_feedback.txt')
    LN_Functions.LME(cohort['data'][cohort['data'].EncIdx == 1], 'EgoRetRankedPerformance', 'TimeSinceEnc', cohort['path'] + 'ego_feedback.txt')
    
# Figure 7B
# Models and p values 
model_allo_on_allo = LN_Functions.LME(cohorts_1and2, 'AlloRetRankedPerformance', 'AlloWithAlloFeedback')
model_allo_on_ego = LN_Functions.LME(cohorts_1and2, 'EgoRetRankedPerformance', 'EgoWithAlloFeedback')
model_ego_on_allo = LN_Functions.LME(cohorts_1and2, 'AlloRetRankedPerformance', 'AlloWithEgoFeedback')
model_ego_on_ego = LN_Functions.LME(cohorts_1and2, 'EgoRetRankedPerformance', 'EgoWithEgoFeedback')
p_allo_on_allo = model_allo_on_allo.pvalues['AlloWithAlloFeedback[T.True]']
p_allo_on_ego = model_allo_on_ego.pvalues['EgoWithAlloFeedback[T.True]']
p_ego_on_allo = model_ego_on_allo.pvalues['AlloWithEgoFeedback[T.True]']
p_ego_on_ego = model_ego_on_ego.pvalues['EgoWithEgoFeedback[T.True]']

# Put together p values for the plot
significance_combinations = [[((1, 2), p_allo_on_allo)],  [((1, 2), p_allo_on_ego)],  [((1, 2), p_ego_on_allo)], [((1, 2), p_ego_on_ego)]]

# Create figure 7B
LN_Figures.figure7b_feedback(cohorts_1and2, significance_combinations, paths['figures'] + 'Figure7B_20250116.svg')

# Figure 7C
means_allo_cohort1 = LN_Functions.calculate_means_allo(cohort1)
means_allo_cohort2 = LN_Functions.calculate_means_allo(cohort2)
means_allo_combined = LN_Functions.calculate_means_allo(cohorts_1and2)
LN_Figures.figure7c_retrieval_position_allo(means_allo_cohort1, means_allo_cohort2, means_allo_combined, paths['figures'] + 'Figure7C_20250116.svg')

# Figure 7D
means_ego_cohort1 = LN_Functions.calculate_means_ego(cohort1)
means_ego_cohort2 = LN_Functions.calculate_means_ego(cohort2)
means_ego_combined = LN_Functions.calculate_means_ego(cohorts_1and2)
LN_Figures.figure7d_retrieval_position_allo(means_ego_cohort1, means_ego_cohort2, means_ego_combined, paths['figures'] + 'Figure7D_20250116.svg')