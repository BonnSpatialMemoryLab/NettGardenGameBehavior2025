"""
This script analyses the behavioral information from the Garden Game task. 
A text file including all information is generated. 
Figures 4A-E are created with this script.

*Object-specific learning*

Laura Nett, 2024
"""
# Imports
import numpy as np
import pandas as pd
import sys
from scipy.stats import ttest_ind, chi2_contingency
import matplotlib.pyplot as plt
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

# Object specific learning
outputfile_osl = paths['results_cohort1and2'] + 'object_specific_learning.txt'
with open(outputfile_osl, 'w') as f:
    f.write("")  # Clear the file content before running analysis

# Get table with information about fitted Weibull function and change points
weibull_table = LN_Functions.create_weibull_table(cohorts_1and2)
weibull_table = LN_Functions.get_change_points(cohorts_1and2, weibull_table)

# Masks
is_allo = (weibull_table.ret_type == 'allo')
is_ego = (weibull_table.ret_type == 'ego')
is_stable = (weibull_table.object_stability == True)

# Dataframes for each category
table_allo = weibull_table[is_allo]
table_ego = weibull_table[is_ego]
table_allo_stable = weibull_table[is_allo & is_stable]
table_allo_unstable = weibull_table[is_allo & ~is_stable]
table_ego_stable = weibull_table[is_ego & is_stable]
table_ego_unstable = weibull_table[is_ego & ~is_stable]

# Number of change points
num_cp_allo = np.sum(table_allo.is_change_point)
num_cp_ego = np.sum(table_ego.is_change_point)
num_cp_allo_stable = np.sum(table_allo_stable.is_change_point)
num_cp_allo_unstable = np.sum(table_allo_unstable.is_change_point)
num_cp_ego_stable = np.sum(table_ego_stable.is_change_point)
num_cp_ego_unstable = np.sum(table_ego_unstable.is_change_point)

# Chi-squared test between number of change points
chi2_allo_ego = chi2_contingency(np.array([np.bincount(table_allo.is_change_point.astype(int)), np.bincount(table_ego.is_change_point.astype(int))]))
chi2_allo_stable_unstable = chi2_contingency(np.array([np.bincount(table_allo_stable.is_change_point.astype(int)), np.bincount(table_allo_unstable.is_change_point.astype(int))]))
chi2_ego_stable_unstable = chi2_contingency(np.array([np.bincount(table_ego_stable.is_change_point.astype(int)), np.bincount(table_ego_unstable.is_change_point.astype(int))]))

# t-test between the initial performances
t_allo_ego_initial = ttest_ind(table_allo.initial_performance, table_ego.initial_performance)
t_allo_stable_unstable_initial = ttest_ind(table_allo_stable.initial_performance, table_allo_unstable.initial_performance)
t_ego_stable_unstable_initial = ttest_ind(table_ego_stable.initial_performance, table_ego_unstable.initial_performance)

# t-test between the asymptotic performances
t_allo_ego_asymptote = ttest_ind(table_allo.c.astype(float), table_ego.c.astype(float))
t_allo_stable_unstable_asymptote = ttest_ind(table_allo_stable.c.astype(float), table_allo_unstable.c.astype(float))
t_ego_stable_unstable_asymptote = ttest_ind(table_ego_stable.c.astype(float), table_ego_unstable.c.astype(float))

# t-test between the trial indices with the change point
t_allo_ego_cp = ttest_ind(table_allo[table_allo.is_change_point].change_point, table_ego[table_ego.is_change_point].change_point)
t_allo_stable_unstable_cp = ttest_ind(table_allo_stable[table_allo_stable.is_change_point].change_point, table_allo_unstable[table_allo_unstable.is_change_point].change_point)
t_ego_stable_unstable_cp = ttest_ind(table_ego_stable[table_ego_stable.is_change_point].change_point, table_ego_unstable[table_ego_unstable.is_change_point].change_point)

# t-test between the slopes
t_allo_ego_slope = ttest_ind(table_allo[table_allo.is_change_point].slope, table_ego[table_ego.is_change_point].slope)
t_allo_stable_unstable_slope = ttest_ind(table_allo_stable[table_allo_stable.is_change_point].slope, table_allo_unstable[table_allo_unstable.is_change_point].slope)
t_ego_stable_unstable_slope = ttest_ind(table_ego_stable[table_ego_stable.is_change_point].slope, table_ego_unstable[table_ego_unstable.is_change_point].slope)

# Write all results to the file
with open(outputfile_osl, 'w') as f:
    f.write(f"Total number of change points in allocentric learning curves: {num_cp_allo}\n")
    f.write(f"Total number of change points in egocentric learning curves: {num_cp_ego}\n")
    f.write(f"Total number of change points for stable animals regarding allocentric learning curves: {num_cp_allo_stable}\n")
    f.write(f"Total number of change points for unstable animals regarding allocentric learning curves: {num_cp_allo_unstable}\n")
    f.write(f"Total number of change points for stable animals regarding egocentric learning curves: {num_cp_ego_stable}\n")
    f.write(f"Total number of change points for unstable animals regarding egocentric learning curves: {num_cp_ego_unstable}\n")
    f.write("\n")
    
    f.write(LN_Functions.format_chi2_result("the number of change points for egocentric vs. allocentric learning curves", chi2_allo_ego))
    f.write(LN_Functions.format_chi2_result("the number of change points for stable vs. unstable animals regarding allocentric learning curves", chi2_allo_stable_unstable))
    f.write(LN_Functions.format_chi2_result("the number of change points for stable vs. unstable animals regarding egocentric learning curves", chi2_ego_stable_unstable))
    f.write("\n")

    f.write(LN_Functions.format_ttest_result("the initial performance for stable vs. unstable animals regarding allocentric learning curves", t_allo_ego_initial))
    f.write(LN_Functions.format_ttest_result("the initial performance for stable vs. unstable animals regarding egocentric learning curves", t_allo_stable_unstable_initial))
    f.write(LN_Functions.format_ttest_result("the initial performance for stable vs. unstable animals regarding egocentric learning curves", t_ego_stable_unstable_initial))
    f.write("\n")

    f.write(LN_Functions.format_ttest_result("the asymptotic performance for allocentric vs. egocentric learning curves", t_allo_ego_asymptote))
    f.write(LN_Functions.format_ttest_result("the asymptotic performance for stable vs. unstable animals regarding allocentric learning curves", t_allo_stable_unstable_asymptote))
    f.write(LN_Functions.format_ttest_result("the asymptotic performance for stable vs. unstable animals regarding egocentric learning curves", t_ego_stable_unstable_asymptote))
    f.write("\n")

    f.write(LN_Functions.format_ttest_result("the trial index with the change point for allocentric vs. egocentric learning curves", t_allo_ego_cp))
    f.write(LN_Functions.format_ttest_result("the trial index with the change point for stable vs. unstable animals regarding allocentric learning curves", t_allo_stable_unstable_cp))
    f.write(LN_Functions.format_ttest_result("the trial index with the change point for stable vs. unstable animals regarding egocentric learning curves", t_ego_stable_unstable_cp))
    f.write("\n")

    f.write(LN_Functions.format_ttest_result("the slope at the change point for allocentric vs. egocentric learning curves", t_allo_ego_slope))
    f.write(LN_Functions.format_ttest_result("the slope at the change point for stable vs. unstable animals regarding allocentric learning curves", t_allo_stable_unstable_slope))
    f.write(LN_Functions.format_ttest_result("the slope at the change point for stable vs. unstable animals regarding egocentric learning curves", t_ego_stable_unstable_slope))

# Figure 4A
fig, axes = plt.subplots(2, 4, figsize=(27, 12), sharey=True)

# First row: Allocentric
LN_Figures.figure4a_weibull_fit(cohorts_1and2, weibull_table, subject=18, animal='Horse', ret_type='allo', ax=axes[0, 0], is_first_column=True) # abrupt learning
LN_Figures.figure4a_weibull_fit(cohorts_1and2, weibull_table, subject=17, animal='Pig', ret_type='allo', ax=axes[0, 1]) # gradual learning
LN_Figures.figure4a_weibull_fit(cohorts_1and2, weibull_table, subject=57, animal='Chicken', ret_type='allo', ax=axes[0, 2]) # no learning with high performance
LN_Figures.figure4a_weibull_fit(cohorts_1and2, weibull_table, subject=5, animal='Camel', ret_type='allo', ax=axes[0, 3]) # no learning with poor performance

# Second row: Egocentric
LN_Figures.figure4a_weibull_fit(cohorts_1and2, weibull_table, subject=11, animal='Tiger', ret_type='ego', ax=axes[1, 0], is_first_column=True) # abrupt learning
LN_Figures.figure4a_weibull_fit(cohorts_1and2, weibull_table, subject=48, animal='Tiger', ret_type='ego', ax=axes[1, 1]) # gradual learning
LN_Figures.figure4a_weibull_fit(cohorts_1and2, weibull_table, subject=46, animal='Cat', ret_type='ego', ax=axes[1, 2]) # no learning with high performance
LN_Figures.figure4a_weibull_fit(cohorts_1and2, weibull_table, subject=17, animal='Chicken', ret_type='ego', ax=axes[1, 3]) # no learning with poor performance

# Add row labels (Allocentric/Egocentric performance)
axes[0, 0].set_ylabel('Allocentric performance', fontsize=30)
axes[1, 0].set_ylabel('Egocentric performance', fontsize=30)
axes[0, 0].set_xlabel('Trial index', fontsize=30)
axes[1, 0].set_xlabel('Trial index', fontsize=30)

# Add titles 
axes[0, 0].set_title('Participant 18, Horse', fontsize=32, pad = 50)
axes[0, 1].set_title('Participant 17, Pig', fontsize=32, pad = 50)
axes[0, 2].set_title('Participant 57, Chicken', fontsize=32, pad = 50)
axes[0, 3].set_title('Participant 5, Camel', fontsize=32, pad = 50)
axes[1, 0].set_title('Participant 11, Tiger', fontsize=32, pad = 50)
axes[1, 1].set_title('Participant 48, Tiger', fontsize=32, pad = 50)
axes[1, 2].set_title('Participant 46, Cat', fontsize=32, pad = 50)
axes[1, 3].set_title('Participant 17, Chicken', fontsize=32, pad = 50)

# Remove x and y-axis labels for non-first-column plots
for row in axes:
    for ax in row[1:]:
        ax.set_ylabel('')
    for ax in row[1:]:
        ax.set_xlabel('')

# Adjust layout
plt.tight_layout()
plt.savefig(paths['figures'] + 'Figure4A_20250116.svg', dpi = 300)

# Figure 4B
bin_edges = [i * 0.05 for i in range(21)]
x_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
y_ticks = [0, 10, 20, 30, 40]
allo_stable = table_allo_stable.initial_performance
allo_unstable = table_allo_unstable.initial_performance
ego_stable = table_ego_stable.initial_performance
ego_unstable = table_ego_unstable.initial_performance
LN_Figures.figure4BCDE_hist_learning(allo_stable, allo_unstable, ego_stable, ego_unstable, bin_edges, x_ticks, y_ticks, 'Initial performance',paths['figures'] + 'Figure4B_20250116.svg')

# Figure 4C
bin_edges = np.linspace(0.4, 1, 21)
x_ticks = [0.4, 0.6, 0.8, 1.0]
y_ticks = [0, 10, 20, 30]
allo_stable = table_allo_stable.c.astype(float)
allo_unstable = table_allo_unstable.c.astype(float)
ego_stable = table_ego_stable.c.astype(float)
ego_unstable = table_ego_unstable.c.astype(float)
LN_Figures.figure4BCDE_hist_learning(allo_stable, allo_unstable, ego_stable, ego_unstable, bin_edges, x_ticks, y_ticks, 'Asymptotic performance',paths['figures'] + 'Figure4C_20250116.svg')

# Figure 4D
bin_edges = [0, 1, 2, 3, 4, 5, 6, 7]
x_ticks = [0, 1, 2, 3, 4, 5, 6, 7]
y_ticks = [0, 10, 20, 30, 40, 50]
allo_stable = table_allo_stable[table_allo_stable.is_change_point].change_point
allo_unstable = table_allo_unstable[table_allo_unstable.is_change_point].change_point
ego_stable = table_ego_stable[table_ego_stable.is_change_point].change_point
ego_unstable = table_ego_unstable[table_ego_unstable.is_change_point].change_point
LN_Figures.figure4BCDE_hist_learning(allo_stable, allo_unstable, ego_stable, ego_unstable, bin_edges, x_ticks, y_ticks, 'Trial with change point',paths['figures'] + 'Figure4D_20250116.svg')

# Figure 4E
bin_edges = [i * 0.015 for i in range(21)]
x_ticks = [0, 0.1, 0.2, 0.3]
y_ticks = [0, 5, 10, 15, 20]
allo_stable = table_allo_stable[table_allo_stable.is_change_point].slope
allo_unstable = table_allo_unstable[table_allo_unstable.is_change_point].slope
ego_stable = table_ego_stable[table_ego_stable.is_change_point].slope
ego_unstable = table_ego_unstable[table_ego_unstable.is_change_point].slope
LN_Figures.figure4BCDE_hist_learning(allo_stable, allo_unstable, ego_stable, ego_unstable, bin_edges, x_ticks, y_ticks, 'Slope at change point',paths['figures'] + 'Figure4E_20250116.svg')

