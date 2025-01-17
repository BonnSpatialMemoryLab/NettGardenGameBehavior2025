"""
This script analyses the behavioral information from the Garden Game task. 
A text file including all information is generated. 
Figure 8B is created with this script.

*Associations between viewing behavior and memory performance*

Laura Nett, 2024
"""

# Imports
import numpy as np
import pandas as pd
import sys
import os
path_to_functions = '..\Functions_20250106'
sys.path.append(path_to_functions)
import LN_Functions_20240912 as LN_Functions
import LN_Figures_20241219 as LN_Figures

# Paths to get/save data
paths = {'cohort2' : 'D:\Publications\GardenGameBehavior\Data\PreProcessing\periods_complete_analysis_cohort2.csv',
         'results_cohort2' : 'D:\Publications\GardenGameBehavior\Results\Cohort2\\',
         'figures' : 'D:\Publications\GardenGameBehavior\Figures\MainFigures\\'}

# Get dataframe and remove practice trial for all analyses
cohort2 = pd.read_csv(paths['cohort2'])
cohort2 = cohort2[cohort2.TrialIdx > 0]

outputfile_allo = paths['results_cohort2'] + 'allo_eye.txt'
outputfile_ego = paths['results_cohort2'] + 'ego_eye.txt'
with open(outputfile_allo, 'w') as f:
    f.write("")  # Clear the file content before running analysis
with open(outputfile_ego, 'w') as f:
    f.write("")  # Clear the file content before running analysis

# Influence of viewing time at the fences
LN_Functions.LME(cohort2, 'AlloRetRankedPerformance', 'EyeEncFence', outputfile_allo) 
LN_Functions.LME(cohort2, 'EgoRetRankedPerformance', 'EyeEncFence', outputfile_ego)

# Influence of viewing time at the north fence
LN_Functions.LME(cohort2, 'AlloRetRankedPerformance', 'EyeEncNorthFence', outputfile_allo) 
LN_Functions.LME(cohort2, 'EgoRetRankedPerformance', 'EyeEncNorthFence', outputfile_ego) 

# Influence of viewing time at the east fence
LN_Functions.LME(cohort2, 'AlloRetRankedPerformance', 'EyeEncEastFence', outputfile_allo) 
LN_Functions.LME(cohort2, 'EgoRetRankedPerformance', 'EyeEncEastFence', outputfile_ego)

# Influence of viewing time at the south fence
LN_Functions.LME(cohort2, 'AlloRetRankedPerformance', 'EyeEncSouthFence', outputfile_allo) 
LN_Functions.LME(cohort2, 'EgoRetRankedPerformance', 'EyeEncSouthFence', outputfile_ego)

# Influence of viewing time at the west fence
LN_Functions.LME(cohort2, 'AlloRetRankedPerformance', 'EyeEncWestFence', outputfile_allo)
LN_Functions.LME(cohort2, 'EgoRetRankedPerformance', 'EyeEncWestFence', outputfile_ego)

# Influence of gaze time at the corners
LN_Functions.LME(cohort2, 'AlloRetRankedPerformance', 'EyeEncCorner', outputfile_allo) 
LN_Functions.LME(cohort2, 'EgoRetRankedPerformance', 'EyeEncCorner', outputfile_ego) 

# Influence of viewing time at the north east corner
LN_Functions.LME(cohort2, 'AlloRetRankedPerformance', 'EyeEncNorthEastCorner', outputfile_allo) 
LN_Functions.LME(cohort2, 'EgoRetRankedPerformance', 'EyeEncNorthEastCorner', outputfile_ego) 

# Influence of viewing time at the south east corner
LN_Functions.LME(cohort2, 'AlloRetRankedPerformance', 'EyeEncSouthEastCorner', outputfile_allo) 
LN_Functions.LME(cohort2, 'EgoRetRankedPerformance', 'EyeEncSouthEastCorner', outputfile_ego) 

# Influence of viewing time at the south west corner
LN_Functions.LME(cohort2, 'AlloRetRankedPerformance', 'EyeEncSouthWestCorner', outputfile_allo) 
LN_Functions.LME(cohort2, 'EgoRetRankedPerformance', 'EyeEncSouthWestCorner', outputfile_ego) 

# Influence of viewing time at the north west corner
LN_Functions.LME(cohort2, 'AlloRetRankedPerformance', 'EyeEncNorthWestCorner', outputfile_allo)
LN_Functions.LME(cohort2, 'EgoRetRankedPerformance', 'EyeEncNorthWestCorner', outputfile_ego)

# Influence of viewing time at the animal
LN_Functions.LME(cohort2, 'AlloRetRankedPerformance', 'EyeEncAnimal', outputfile_allo)
LN_Functions.LME(cohort2, 'EgoRetRankedPerformance', 'EyeEncAnimal', outputfile_ego) 

# Influence of viewing time at the gaze area
LN_Functions.LME(cohort2, 'AlloRetRankedPerformance', 'EyeEncGazeArea', outputfile_allo) 
LN_Functions.LME(cohort2, 'EgoRetRankedPerformance', 'EyeEncGazeArea', outputfile_ego) 

# Influence of viewing time at the gaze are and the animal
LN_Functions.LME(cohort2, 'AlloRetRankedPerformance', 'EyeEncGazeAreaAndAnimal', outputfile_allo)
LN_Functions.LME(cohort2, 'EgoRetRankedPerformance', 'EyeEncGazeAreaAndAnimal', outputfile_ego)

# Influence of gaze coverage of the environment
LN_Functions.LME(cohort2, 'AlloRetRankedPerformance', 'EyeEncCoverage', outputfile_allo) 
LN_Functions.LME(cohort2, 'EgoRetRankedPerformance', 'EyeEncCoverage', outputfile_ego)

# Figure 8B
LN_Figures.figure8b_explanation_gaze_area(paths['figures'] + "Figure8B_20250116.svg")