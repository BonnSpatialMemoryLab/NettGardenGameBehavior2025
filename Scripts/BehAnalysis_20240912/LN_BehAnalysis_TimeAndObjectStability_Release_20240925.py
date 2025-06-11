"""
This script analyses the behavioral information from the Garden Game task. 
For each group a text file including all information is generated. 
Figure 3B is created with this script.

*Associations of time and object stability with memory performance*

Laura Nett, 2024
"""

# Imports
import pandas as pd
import sys
import os
from scipy.stats import ttest_rel, pearsonr
path_to_functions = '..\Functions_20250106'
sys.path.append(path_to_functions)
import LN_Functions_Release_20240912 as LN_Functions
import LN_Figures_Release_20241219 as LN_Figures


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

# Influence of time and object stability
for cohort in cohort_data:
    outputfile_allo = cohort['path'] + 'allo_time_and_stability.txt'
    outputfile_ego = cohort['path'] + 'ego_time_and_stability.txt'
    with open(outputfile_allo, 'w') as f:
        f.write("")  # Clear the file content before running analysis
    with open(outputfile_ego, 'w') as f:
        f.write("")  # Clear the file content before running analysis
        
    # Mean allocentric and egocentric performance scores for each subject
    allo_perf = cohort['data'].groupby(['Subject'])['AlloRetRankedPerformance'].mean()
    ego_perf = cohort['data'].groupby(['Subject'])['EgoRetRankedPerformance'].mean()
    
    with open(outputfile_allo, 'w') as f:
        # Paired t-test between allocentric and egocentric memory performance
        f.write("Paired t-test between allocentric and egocentric memory performance:\n")
        f.write(f"t = {round(ttest_rel(allo_perf, ego_perf)[0], 3)}, ")
        f.write(f"p = {ttest_rel(allo_perf, ego_perf)[1]:.3e}\n\n")

        # Pearson correlation between allocentric and egocentric memory performance
        f.write("Pearson correlation between allocentric and egocentric memory performance:\n")
        f.write(f"t = {round(pearsonr(allo_perf, ego_perf)[0], 3)}, ")
        f.write(f"p = {pearsonr(allo_perf, ego_perf)[1]:.3e}\n")
    
    with open(outputfile_ego, 'w') as f:
        # Paired t-test between allocentric and egocentric memory performance
        f.write("Paired t-test between allocentric and egocentric memory performance:\n")
        f.write(f"t = {round(ttest_rel(allo_perf, ego_perf)[0], 3)}, ")
        f.write(f"p = {ttest_rel(allo_perf, ego_perf)[1]:.3e}\n\n")

        # Pearson correlation between allocentric and egocentric memory performance
        f.write("Pearson correlation between allocentric and egocentric memory performance:\n")
        f.write(f"t = {round(pearsonr(allo_perf, ego_perf)[0], 3)}, ")
        f.write(f"p = {pearsonr(allo_perf, ego_perf)[1]:.3e}\n")
        
    # Allocentric
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'TrialIdx', outputfile_allo)
    LN_Functions.LME(cohort['data'][cohort['data'].StableObj == True], 'AlloRetRankedPerformance', 'TrialIdx', outputfile_allo)
    LN_Functions.LME(cohort['data'][cohort['data'].StableObj == False], 'AlloRetRankedPerformance', 'TrialIdx', outputfile_allo)
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'StableObj', outputfile_allo)
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'StableObj + DistObjNearestFence', outputfile_allo)
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'TrialIdx * StableObj', outputfile_allo, mean_centered = ['TrialIdx'])
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'TrialIdx * StableObj + DistObjNearestFence', outputfile_allo, mean_centered = ['TrialIdx'])
    
    
    # Egocentric
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'TrialIdx', outputfile_ego)
    LN_Functions.LME(cohort['data'][cohort['data'].StableObj == True], 'EgoRetRankedPerformance', 'TrialIdx', outputfile_ego)
    LN_Functions.LME(cohort['data'][cohort['data'].StableObj == False], 'EgoRetRankedPerformance', 'TrialIdx', outputfile_ego)
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'StableObj', outputfile_ego)
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'StableObj + DistObjNearestFence', outputfile_ego)
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'TrialIdx * StableObj', outputfile_ego, mean_centered = ['TrialIdx'])
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'TrialIdx * StableObj + DistObjNearestFence', outputfile_ego, mean_centered = ['TrialIdx'])
    
    
    # Interaction between retrieval type and object stability (dependent variable is performance)
    # Add columns performance and retrieval type
    cohort['data'].loc[cohort['data'].PeriodType == 'allocentric retrieval','Performance'] = cohort['data'].loc[cohort['data'].PeriodType == 'allocentric retrieval','AlloRetRankedPerformance']
    cohort['data'].loc[cohort['data'].PeriodType == 'egocentric retrieval','Performance'] = cohort['data'].loc[cohort['data'].PeriodType == 'egocentric retrieval','EgoRetRankedPerformance']
    cohort['data'].loc[cohort['data'].PeriodType == 'allocentric retrieval','RetType'] = 'allo'
    cohort['data'].loc[cohort['data'].PeriodType == 'egocentric retrieval','RetType'] = 'ego'
    LN_Functions.LME(cohort['data'], 'Performance', 'RetType * StableObj', outputfile_allo)
    LN_Functions.LME(cohort['data'], 'Performance', 'RetType * StableObj', outputfile_ego)
    

# Figure 3B
# Models and p values allocentric
model_allo_cohort1 = LN_Functions.LME(cohort1, 'AlloRetRankedPerformance', 'StableObj')
model_allo_cohort2 = LN_Functions.LME(cohort2, 'AlloRetRankedPerformance', 'StableObj')
model_allo_cohorts_1and2 = LN_Functions.LME(cohorts_1and2, 'AlloRetRankedPerformance', 'StableObj')
p_allo_cohort1 = model_allo_cohort1.pvalues['StableObj[T.True]']
p_allo_cohort2 = model_allo_cohort2.pvalues['StableObj[T.True]']
p_allo_cohorts1and2 = model_allo_cohorts_1and2.pvalues['StableObj[T.True]']

# Models and p values egocentric
model_ego_cohort1 = LN_Functions.LME(cohort1, 'EgoRetRankedPerformance', 'StableObj')
model_ego_cohort2 = LN_Functions.LME(cohort2, 'EgoRetRankedPerformance', 'StableObj')
model_ego_cohorts_1and2 = LN_Functions.LME(cohorts_1and2, 'EgoRetRankedPerformance', 'StableObj')
p_ego_cohort1 = model_ego_cohort1.pvalues['StableObj[T.True]']
p_ego_cohort2 = model_ego_cohort2.pvalues['StableObj[T.True]']
p_ego_cohorts1and2 = model_ego_cohorts_1and2.pvalues['StableObj[T.True]']

# Put together p values for the plot
significance_combinations = [[ ((1,2), p_allo_cohort1), ((3,4), p_ego_cohort1)],[ ((1,2), p_allo_cohort2), ((3,4), p_ego_cohort2)],[ ((1,2), p_allo_cohorts1and2), ((3,4), p_ego_cohorts1and2)]]

# Create figure 3B
LN_Figures.figure3b_influence_object_stability(cohort1, cohort2, cohorts_1and2, significance_combinations, paths['figures'] + 'Figure3B_20250116.svg')