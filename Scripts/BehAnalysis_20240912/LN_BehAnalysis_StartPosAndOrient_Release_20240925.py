"""
This script analyses the behavioral information from the Garden Game task. 
For each group a text file including all information is generated. 
Figures 6C-F are created with this script.

*Associations between starting positions and memory performance*

Laura Nett, 2024
"""

# Imports
import pandas as pd
import scipy
import sys
import os
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


# Influence of starting position and orientation
for cohort in cohort_data:
    outputfile_allo = cohort['path'] + 'allo_start_pos_and_orient.txt'
    outputfile_ego = cohort['path'] + 'ego_start_pos_and_orient.txt'
    with open(outputfile_allo, 'w') as f:
        f.write("")  # Clear the file content before running analysis
    with open(outputfile_ego, 'w') as f:
        f.write("")  # Clear the file content before running analysis
        
    # Distance to starting position
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'DistObjPlayerStart', outputfile_allo)    
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'DistObjPlayerStart', outputfile_ego) 
    
    # Alignment with cardinal axes
    LN_Functions.LME(cohort['data'], 'AlloRetRankedPerformance', 'AlloStartPosAligned', outputfile_allo) 
    LN_Functions.LME(cohort['data'], 'EgoRetRankedPerformance', 'AlloStartPosAligned', outputfile_ego) 
    
    # Friedman test for starting orientation
    LN_Functions.friedman_test_influence_orientation(cohort['data'], 'AlloRetRankedPerformance', 'AlloStartPosOrient8Bins', outputfile_allo)
    LN_Functions.friedman_test_influence_orientation(cohort['data'], 'AlloRetRankedPerformance', 'AlloStartPosOrient12Bins', outputfile_allo)
    LN_Functions.friedman_test_influence_orientation(cohort['data'], 'AlloRetRankedPerformance', 'EgoStartPosOrient', outputfile_allo)
    LN_Functions.friedman_test_influence_orientation(cohort['data'], 'EgoRetRankedPerformance', 'AlloStartPosOrient8Bins', outputfile_ego)
    LN_Functions.friedman_test_influence_orientation(cohort['data'], 'EgoRetRankedPerformance', 'AlloStartPosOrient12Bins', outputfile_ego)
    LN_Functions.friedman_test_influence_orientation(cohort['data'], 'EgoRetRankedPerformance', 'EgoStartPosOrient', outputfile_ego)
    
# Figure 6C
LN_Figures.figures6cdef_plot_influence_starting_orientation(cohort1 = cohort1, cohort2 = cohort2, cohort3 = cohorts_1and2,
                                    performance ='AlloRetRankedPerformance', 
                                    orientation ='EgoStartPosOrient', 
                                    y_label='Allocentric performance', 
                                    x_label='Egocentric direction of object', 
                                    title1='Cohort 1', title2='Cohort 2', title3='Cohorts 1 and 2',
                                    ylim = (0.25,1), y_ticks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  
                                    y_ticklabels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
                                    custom_order=['B', 'BL', 'LB', 'L', 'LA', 'AL', 'A', 'AR', 'RA', 'R', 'RB', 'BR'], 
                                    cardinal_directions={0: 'B', 3: 'L', 6: 'A', 9: 'R'}, 
                                    colors = ['deepskyblue', 'cornflowerblue', 'royalblue'], 
                                    outputfile = paths['figures'] + 'Figure6C_20250116.svg')

# Figure 6D
LN_Figures.figures6cdef_plot_influence_starting_orientation(cohort1 = cohort1, cohort2 = cohort2, cohort3 = cohorts_1and2,
                                    performance ='EgoRetRankedPerformance', 
                                    orientation ='EgoStartPosOrient', 
                                    y_label='Egocentric performance', 
                                    x_label='Egocentric direction of object', 
                                    title1='Cohort 1', title2='Cohort 2', title3='Cohorts 1 and 2',
                                    ylim = (0.25,1), y_ticks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  
                                    y_ticklabels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
                                    custom_order=['B', 'BL', 'LB', 'L', 'LA', 'AL', 'A', 'AR', 'RA', 'R', 'RB', 'BR'], 
                                    cardinal_directions={0: 'B', 3: 'L', 6: 'A', 9: 'R'}, 
                                    colors=['limegreen', '#1f9b1f', '#117711'], 
                                    outputfile = paths['figures'] + 'Figure6D_20250116.svg')

# Figure 6E
LN_Figures.figures6cdef_plot_influence_starting_orientation(cohort1 = cohort1, cohort2 = cohort2, cohort3 = cohorts_1and2,
                                    performance ='AlloRetRankedPerformance', 
                                    orientation ='AlloStartPosOrient', 
                                    y_label = 'Allocentric performance', 
                                    x_label ='Allocentric direction of starting position', 
                                    title1 = 'Cohort 1', title2 = 'Cohort 2', title3 ='Cohorts 1 and 2',
                                    ylim = (0.25,1), y_ticks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  
                                    y_ticklabels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
                                    custom_order = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE'], 
                                    cardinal_directions = {0: 'S', 2: 'W', 4: 'N', 6: 'E'}, 
                                    colors = ['deepskyblue', 'cornflowerblue', 'royalblue'], 
                                    outputfile = paths['figures'] + 'Figure6E_20250116.svg')

# Figure 6F
LN_Figures.figures6cdef_plot_influence_starting_orientation(cohort1 = cohort1, cohort2 = cohort2, cohort3 = cohorts_1and2,
                                    performance ='EgoRetRankedPerformance', 
                                    orientation ='AlloStartPosOrient', 
                                    y_label = 'Egocentric performance', 
                                    x_label ='Allocentric direction of starting position', 
                                    title1 = 'Cohort 1', title2 = 'Cohort 2', title3 ='Cohorts 1 and 2',
                                    ylim = (0.25,1), y_ticks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  
                                    y_ticklabels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  
                                    custom_order = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE'], 
                                    cardinal_directions = {0: 'S', 2: 'W', 4: 'N', 6: 'E'}, 
                                    colors=['limegreen', '#1f9b1f', '#117711'], 
                                    outputfile = paths['figures'] + 'Figure6F_20250116.svg')

