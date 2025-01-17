# Behavioral investigation of allocentric and egocentric cognitive maps in human spatial memory

### Contents
- [Overview](#overview)
- [Software requirements](#requirements)
- [Code and Data structure](#code-and-data-structure)
- [Installation](#installation)
- [How to use the code](#how-to-use-the-code)

### Overview
This repository contains codes and data associated with the paper *Behavioral investigation of allocentric and egocentric cognitive maps in human spatial memory*.

### Software requirements
Most codes are written in python (version 3.11.5). Regression plots are generated using R.

The following python packages are required:
* *numpy*
* *pandas*
* *scipy*
* *statsmodels*
* *matplotlib*
* *seaborn*
* *pingouin*

The following R packages are required:
* *ggplot2*
* *lme4*
* *dplyr*
* *effects*
* *patchwork*
* *svglite*

### Code and Data structure

#### Folder Structure

- **Data**
- **Results**
  - **Add**
  - **Cohort1**
  - **Cohort2**
  - **Cohorts1and2**
  - **Complete**
- **Figures**
- **Scripts**
  - **Functions_20250106**
  - **BehPreProcessing_20231017**
  - **BehAnalysis_20240912**

This repository is organized into four main directories: **Data**, **Results**, **Figures**, and **Scripts**.

- The **Data** directory contains a link to the dataset hosted on Zenodo.
- The **Results** directory includes the outcomes of all analyses conducted in the Garden Game behavioral study, with each subgroup and analysis having its own dedicated text file.
- The **Scripts** directory holds all the scripts used for data preprocessing and analysis in the Garden Game study. Each script loads data from the **Data** directory, saves the results in the **Results** folder, and generates figures in the **Figures** folder.

### Installation
Download or clone this repository:

```bash
git clone https://github.com/BonnSpatialMemoryLab/NettGardenGameBehavior2025.git
cd NettGardenGameBehavior2025
```

### How to use the code
##### Behavioral preprocessing
1. Navigate to the **Scripts/BehPreProcessing_20231017** folder.
2. There are two scripts available:
   - **LN_BehPreProcessing_PerSubject_Release_20240919**: Preprocesses the behavioral data from the logfile for a particular participant.
   - **LN_BehPreProcessing_PeriodsComplete_Release_20241230**: Creates a large data frame containing all the necessary information for analyzing the behavioral data of the Garden Game.

The script **LN_BehPreProcessing_PerSubject_Release_20240919** generates sanity check figures and the following three dataframes for each subject:
1. **periods**  
   Contains information about the behavioral periods for each subject, such as start and end times, the participant's starting position and orientation, final response location, etc.

2. **timeseries**  
   Includes the time-resolved data for each subject, capturing detailed events over time, such as the participant's positions and orientations during navigation and the participant's response positions during retrieval.

3. **eye**  
    Includes the time-resolved eye tracking data for each subject, capturing the participant's gaze over time.

To generate these data frames, adjust the paths and parameters as needed, then run the following command:
```bash
python LN_BehPreProcessing_PerSubject_Release_20240919.py
```

The script **LN_BehPreProcessing_PeriodsComplete_Release_20241230** generates the following three dataframes for each group:

1. **periods_complete_no_excluded**  
   Combines the period data frames from all subjects and includes additional information for analysis, such as the distance of the animal to the nearest fence.

2. **periods_complete_no_analysis**  
   Similar to **periods_complete_no_excluded**, but excludes trials where one or more recall periods exceed 90 seconds. For these trials, the allocentric and egocentric performance scores (AlloRetRankedPerformance and EgoRetRankedPerformance) are set to "NaN".

3. **periods_complete_analysis**  
   Builds on **periods_complete_no_excluded** by adding information from the encoding periods (e.g., the distance of the animal to the nearest fence) to their corresponding retrieval periods. This makes the data more suitable for use in linear mixed models.

To generate these dataframes, adjust the paths and parameters as needed, then run the following command:

```bash
python LN_BehPreProcessing_PeriodsComplete_Release_20241230.py
```

##### Behavioral analysis
1. Navigate to the **Scripts/BehAnalysis_20240912** folder.
2. The following scripts are available:
   - **LN_BehAnalysis_AgeAndGender_Release_20240925**: Age and Gender effects.
   - **LN_BehAnalysis_BasicBehavior_Release_20250101**: Participant information and basic behavior.
   - **LN_BehAnalysis_EnvFeatures_Release_20240925**: Associations of environmental features with memory performance.
   - **LN_BehAnalysis_Eyetracking_Release_20241022**: Associations between viewing behavior and memory performance.
   - **LN_BehAnalysis_Feedback_Release_20240925**: Associations between feedback and memory performance.
   - **LN_BehAnalysis_ObjSpecLearning_Release_20241219**: Object-specific learning.
   - **LN_BehAnalysis_RegressionPlots_Release_20241230**: Regression plots for all analyses.
   - **LN_BehAnalysis_StartPosAndOrient_Release_20240925**: Associations between starting positions and memory performance.
   - **LN_BehAnalysis_TimeAndObjectStability_Release_20240925**: Associations of time and object stability with memory performance.

To create a text file with the results and the corresponding figures for the analysis of time and object stability, adjust the paths as needed, then run the following command:
```bash
python LN_BehAnalysis_TimeAndObjectStability_Release_20240925.py
```

For creating all regression plots, adjust the paths as needed, then run the following command:
```bash
Rscript LN_RegressionPlots_Release_20241230.r
```

------------------------------------------------------------------------------------------
