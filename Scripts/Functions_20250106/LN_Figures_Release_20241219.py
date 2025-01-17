"""
This script includes functions for creating figures for the behavioral analyses from the Garden Game task.

Laura Nett, 2024
"""

# Imports
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import math
import numpy as np
import pandas as pd
import sys
import seaborn as sns
import LN_Functions_20240912 as LN_Functions

# Update font, so it fits font from Regression plots generated using R
plt.rcParams.update({ 'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial']})

# Parameters needed for figures
params = {'ArenaEdgeLength' : 20,
          'EgoRetRadius' : 10}

# Figure 2A: Duration of each trial period
def figure2A(periods_complete, outputfile):
    cohorts_complete = periods_complete.copy()
    
    # Calculate the duration for each period
    period_duration = (cohorts_complete.PeriodEndTime.values.astype(float) - cohorts_complete.PeriodStartTime.values.astype(float)) / 1000 / 60  # in minutes
    cohorts_complete.loc[:,'PeriodDuration'] = period_duration

    # Group by Subject and PeriodType, then calculate the mean duration for each period type across subjects
    mean_durations = cohorts_complete.groupby(['Subject', 'PeriodType']).PeriodDuration.sum().reset_index()

    # Group by PeriodType to get the mean duration across all subjects
    mean_durations_across_subjects = mean_durations.groupby('PeriodType').PeriodDuration.mean().reset_index()

    # Create the figure
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.spines['top'].set_color('lightgray')
    ax.spines['right'].set_color('lightgray')
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')
    ax.grid(axis='x', zorder=0)  
    ax.set_axisbelow(True)       
    
    period_types = mean_durations_across_subjects['PeriodType'].values
    period_types = np.array([s.capitalize() for s in period_types], dtype='object')

    # Plot the boxplot 
    sns.boxplot(x='PeriodDuration', y='PeriodType', data=mean_durations, color = 'lightgrey')
    plt.xlabel('Total period duration (min)', fontsize=24)
    plt.ylabel('')
    plt.yticks(range(0, len(mean_durations['PeriodType'].unique())),period_types, fontsize=24)
    ax.tick_params(axis='y', which='both', length=0)
    plt.xticks(fontsize=24)
    plt.savefig(outputfile, bbox_inches = 'tight', format = 'svg')
    plt.close()
    
# Figure 2B: Overall task duration  
def figure2B(periods_complete, outputfile):
    # Calculate the period duration for each subject
    period_durations_per_subject = ((periods_complete.groupby('Subject')['PeriodEndTime'].max() - periods_complete.groupby('Subject')['PeriodStartTime'].min()) / 1000 / 60).values

    # Create the figure 
    plt.figure(figsize=(6, 6))
    plt.gca().spines['top'].set_color('lightgray')
    plt.gca().spines['right'].set_color('lightgray')
    plt.gca().spines['left'].set_color('lightgray')
    plt.gca().spines['bottom'].set_color('lightgray')
    plt.grid(axis='y')
    plt.scatter(np.arange(0,len(period_durations_per_subject)), np.sort(period_durations_per_subject), color='blue', s=10)
    plt.xlabel('Participants (sorted)', fontsize=24, labelpad = 20)
    plt.ylabel('Task duration (min)', fontsize=24)
    plt.xticks([]) 
    plt.yticks(fontsize=24)
    plt.gca().tick_params(axis='y', which='both', length=0)
    plt.savefig(outputfile, bbox_inches='tight', format = 'svg')
    plt.close()
    
# Figure 2C: Total memory points per trial
def figure2C(periods_complete, outputfile):
    # Calculate the points earned for each trial
    trial_points = periods_complete.groupby(['Subject', 'TrialIdx']).PlayerScore.last().groupby('Subject').diff().fillna(periods_complete.groupby(['Subject', 'TrialIdx']).PlayerScore.last())

    # Filter to keep only rows where TrialIdx is greater than 0 (remove practice trial)
    trial_points_filtered = trial_points[trial_points.index.get_level_values('TrialIdx') > 0].reset_index()
    pivoted_data = trial_points_filtered.pivot(index='Subject', columns='TrialIdx', values='PlayerScore')
    data = pivoted_data.to_numpy()

    # Calculate percentiles
    median = np.percentile(data, 50, axis=0)
    p25 = np.percentile(data, 25, axis=0)
    p75 = np.percentile(data, 75, axis=0)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    # Create figure
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.spines['top'].set_color('lightgray')
    ax.spines['right'].set_color('lightgray')
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')
    ax.grid(axis='y', zorder=0)   
    ax.set_axisbelow(True) 

    # Shaded area between 25th and 75th percentiles
    plt.fill_between(range(1, data.shape[1] + 1), p25, p75, color='green', alpha=0.3)

    # Plot min and max values as lines
    plt.plot(range(1, data.shape[1] + 1), min_vals, color='blue', linestyle='--', label='Min', alpha=0.7)
    plt.plot(range(1, data.shape[1] + 1), max_vals, color='red', linestyle='--', label='Max', alpha=0.7)

    # Plot median as a thick line
    plt.plot(range(1, data.shape[1] + 1), median, color='green', linewidth=3, label='Median')

    # Customize plot
    plt.xlabel('Trial index', fontsize=24)
    plt.ylabel('Memory points', fontsize=24)
    plt.xlim(1, 60)
    plt.ylim(-1,50)
    plt.legend(fontsize = 22, frameon = False, loc='upper left', ncol=3, handlelength=1, columnspacing=1.0)
    plt.xticks([0,10,20,30,40,50,60], fontsize=22)
    plt.yticks(np.arange(0, 41, 10), fontsize=22)
    ax.tick_params(axis='both', which='both', length=0)
    plt.savefig(outputfile, bbox_inches='tight', format = 'svg') 
    plt.close()

# Figure 2D-G, first column: Allocentric starting positions and orientations for all trials
def figure2_starting_positions_and_orientations(periods, outputfile):
    # Remove practice trial
    periods, _ = LN_Functions.remove_practice_trial(periods, None)
    
    # Calculate starting position and orientation
    is_start_pos = LN_Functions.is_period_type(periods, 'starting position')
    start_xz = np.array([periods.StartPosX[is_start_pos],
                          periods.StartPosZ[is_start_pos]])
    start_yaw = periods.StartPosCartYaw[is_start_pos]
    
    # Arena edge length
    arena_edge_length = params['ArenaEdgeLength']
    
    # Create figure
    fig = plt.figure(figsize=(6, 6))

    # Customize plot
    ax = plt.axes([1.25 / 6, 0.8 / 6, 4 / 6, 4 / 6])
    ax.set_xlim(arena_edge_length * np.array([-0.5, 0.5]))
    ax.set_ylim(arena_edge_length * np.array([-0.5, 0.5]))
    ax.set_xticks(arena_edge_length * np.array([-0.5, 0.5]))
    ax.set_xticklabels((arena_edge_length * np.array([-0.5, 0.5])).astype(int), fontsize=24)
    ax.set_yticks(arena_edge_length * np.array([-0.5, 0.5]))
    ax.set_yticklabels((arena_edge_length * np.array([-0.5, 0.5])).astype(int), fontsize=24)
    ax.set_xlabel('x (vu)', fontsize=26, labelpad=-22)
    ylabel_obj = ax.set_ylabel('z (vu)', rotation = 0, fontsize=26, labelpad = -5)
    ylabel_obj.set_position((ylabel_obj.get_position()[0], ylabel_obj.get_position()[1] - 0.04))
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(3)

    # Plot starting position and orientation
    for i_start in range(start_xz.shape[1]):
        ax.plot(start_xz[0, i_start], start_xz[1, i_start], 'x', color='red', markersize=10)  # Location
        ax.quiver(start_xz[0, i_start], start_xz[1, i_start], np.cos(start_yaw.iloc[i_start]) * 2,
                   np.sin(start_yaw.iloc[i_start]) * 2, linewidth=1, angles='xy', scale_units='xy', scale=1,
                   headwidth=6, headlength=5, color='black')
    ax.set_box_aspect(True)
    plt.title('Starting positions', fontsize=30, pad = 50) 
    fig.savefig(outputfile, format = 'svg')
    plt.close()
    
# Figure 2D-G, second column: Navigation trajectory during all encoding periods
def figure2_enc_trajectory(periods, timeseries, outputfile):
    # Remove practice trial
    periods, timeseries = LN_Functions.remove_practice_trial(periods, timeseries)
    
    # Get encoding periods
    encoding = periods[periods.PeriodType == 'encoding']

    # Get start and stop time of encodings per trial
    enc = encoding.groupby('TrialIdx').agg(
        TrialStartTime=('PeriodStartTime', 'min'),
        TrialEndTime=('PeriodEndTime', 'max')
    ).reset_index()
    
    # Arena edge length
    arena_edge_length = params['ArenaEdgeLength']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_position([1.5 / 6, 0.8 / 6, 4 / 6, 4 / 6])

    # Plot each encoding trajectory
    for i, row in enc.iterrows():
        enc_timeseries = timeseries[(timeseries.Time <= row.TrialEndTime) & (timeseries.Time >= row.TrialStartTime)]
        ax.plot(enc_timeseries.EncPlayerX, enc_timeseries.EncPlayerZ, color=[0.5, 0.5, 0.5], linewidth=0.7)
    
    # Customize plot
    ax.set_xlim(arena_edge_length * np.array([-0.5, 0.5]))
    ax.set_ylim(arena_edge_length * np.array([-0.5, 0.5]))
    ax.set_xticks(arena_edge_length * np.array([-0.5, 0.5]))
    ax.set_xticklabels((arena_edge_length * np.array([-0.5, 0.5])).astype(int), fontsize=24)
    ax.set_yticks(arena_edge_length * np.array([-0.5, 0.5]))
    ax.set_yticklabels((arena_edge_length * np.array([-0.5, 0.5])).astype(int), fontsize=24)
    ax.set_xlabel('x (vu)', fontsize=26, labelpad=-22)
    ylabel_obj = ax.set_ylabel('z (vu)', rotation = 0, fontsize=26, labelpad = -5)
    ylabel_obj.set_position((ylabel_obj.get_position()[0], ylabel_obj.get_position()[1] - 0.04))
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(3)
    ax.set_title('Encoding', fontsize=30, pad = 50) 
    fig.savefig(outputfile, format = 'svg')
    plt.close()

# Figure 2D-G, third column: Cursor trajectory during all allocentric retrieval periods
def figure2_allo_trajectory(periods, timeseries, outputfile):
    # Remove practice trial
    periods, timeseries = LN_Functions.remove_practice_trial(periods, timeseries)
    
    # Get allocentric retrieval periods
    allo_ret = periods[periods.PeriodType == 'allocentric retrieval']

    # Get start and stop time of allocentric retrievals per trial
    allo = allo_ret.groupby('TrialIdx').agg(
        TrialStartTime=('PeriodStartTime', 'min'),
        TrialEndTime=('PeriodEndTime', 'max')
    ).reset_index()
    
    # Arena edge length
    arena_edge_length = params['ArenaEdgeLength']
    
    # Create figure
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes([1.5 / 6, 0.8 / 6, 4 / 6, 4 / 6])
    
    # Plot each allocentric retrieval trajectory
    for i, row in allo.iterrows():
        allo_timeseries = timeseries[(timeseries.Time <= row.TrialEndTime) & (timeseries.Time >= row.TrialStartTime)]
        ax.plot(allo_timeseries.AlloRetPlayerXAlloMap, allo_timeseries.AlloRetPlayerZAlloMap, color=[0, 0, 1], linewidth=0.7)
    
    # Customize plot
    ax.set_xlim(arena_edge_length * np.array([-0.5, 0.5]))
    ax.set_ylim(arena_edge_length * np.array([-0.5, 0.5]))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('', fontsize=26)
    ax.set_ylabel('', fontsize=26)
    ax.set_title('Allocentric retrieval', fontsize=30, pad = 50) 
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(3)
    
    # Add text
    ax.text(0, arena_edge_length * 0.5 + 0.6, 'N', ha='center', va='bottom', fontsize=26, color='black')  
    ax.text(arena_edge_length * 0.5 + 0.7, 0, 'E', ha='left', va='center', fontsize=26, color='black')   
    ax.text(3, -arena_edge_length * 0.5 - 0.7, 'S (20 vu)', ha='center', va='top', fontsize=26, color='black')   
    ax.text(-arena_edge_length * 0.5 - 0.7, -1, 'W\n(20 vu)', ha='right', va='center', fontsize=26, color='black') 

    fig.savefig(outputfile, format = 'svg')
    plt.close()
    
# Figure 2D-G, fourth column: Cursor trajectory during all egocentric retrieval periods
def figure2_ego_trajectory(periods, timeseries, outputfile):
    # Remove practice trial
    periods, timeseries = LN_Functions.remove_practice_trial(periods, timeseries)
    
    # Get egocentric retrieval periods
    ego_ret = periods[periods.PeriodType == 'egocentric retrieval']

    # Get start and stop time of egocentric retrievals per trial
    ego = ego_ret.groupby('TrialIdx').agg(
        TrialStartTime=('PeriodStartTime', 'min'),
        TrialEndTime=('PeriodEndTime', 'max')
    ).reset_index()
    
    # Egocentric radius
    ego_radius = params['EgoRetRadius']
    
    # Create figure
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes([1.5 / 6, 0.8 / 6, 4 / 6, 4 / 6])

    # Plot circles
    theta = np.linspace(0, 2*np.pi, 1000)
    x_circle = np.cos(theta) * ego_radius
    y_circle = np.sin(theta) * ego_radius
    ax.plot(x_circle, y_circle, '.', color=[0.5, 0.5, 0.5], markersize=1)

    # Plot each egocentric retrieval trajectory
    for i, row in ego.iterrows():
        ego_timeseries = timeseries[(timeseries.Time <= row.TrialEndTime) & (timeseries.Time >= row.TrialStartTime)]
        ax.plot(ego_timeseries.EgoRetPlayerXEgoMap, ego_timeseries.EgoRetPlayerZEgoMap, color=[0, 0.5, 0], linewidth=0.7)
    
    # Customize plot
    ax.set_xlim(ego_radius * np.array([-1, 1]) + np.array([-0.1,0.1]))
    ax.set_ylim(ego_radius * np.array([-1, 1]) + np.array([-0.1,0.1]))
    ax.text(0, ego_radius * 1.05, 'A', verticalalignment='bottom', 
            horizontalalignment='center', fontsize=26)  
    ax.text(1.08 * ego_radius, 0, 'R', verticalalignment='center', 
            horizontalalignment='left', fontsize=26)  
    ax.text(0, -ego_radius * 1.08, 'B', verticalalignment='top', 
            horizontalalignment='center',fontsize=26)
    ax.text(-1.08 * ego_radius, 0, 'L', verticalalignment='center', 
            horizontalalignment='right',fontsize=26)
    ax.text(0.95 * ego_radius, 0.8 * ego_radius, '(20 vu)', 
        verticalalignment='center', horizontalalignment='center', fontsize=26)
    ax.set_title('Egocentric retrieval', fontsize=30, pad = 50)
    ax.axis('off')
    fig.savefig(outputfile, format = 'svg')
    plt.close()
    
# Figure 3A: Location of stable and unstable animals across all trials
def figure3A_object_locations(periods, outputfile):
    # Remove practice trial
    periods, _ = LN_Functions.remove_practice_trial(periods, None)

    # Arena edge length
    arena_edge_length = params['ArenaEdgeLength']
    
    # Stable objects
    animals = np.unique(periods[~periods.EncObj.isna()].EncObj)
    stable_objects = LN_Functions.get_stable_objects(periods)
    unstable_objects = animals[~np.isin(animals, stable_objects)]

    # Create figure
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes([1.25 / 6, 1.25 / 6, 4 / 6, 4 / 6])
    ax.set_xlim(arena_edge_length * np.array([-0.5, 0.5]))
    ax.set_ylim(arena_edge_length * np.array([-0.5, 0.5]))
    ax.set_xticks(arena_edge_length * np.array([-0.5, 0.5]))
    ax.set_xticklabels((arena_edge_length * np.array([-0.5, 0.5])).astype(int), fontsize=22)
    ax.set_yticks(arena_edge_length * np.array([-0.5, 0.5]))
    ax.set_yticklabels((arena_edge_length * np.array([-0.5, 0.5])).astype(int), fontsize=22)
    ax.set_xlabel('x (vu)', fontsize=24, labelpad=-22)
    ax.set_ylabel('z (vu)', rotation = 0, fontsize=24, labelpad = -7)

    # Colors for stable/unstable animals
    colors_stable = ['#0072B2', '#009E73', '#BFD3C1']
    colors_unstable = ['#FFDE21', '#FF9800', '#D84315']

    # Plot locations of stable animals
    for i, animal in enumerate(stable_objects):
        # Filter the data for the current object (animal)
        animal_data = periods[periods.EncObj == animal]
        ax.plot(animal_data.EncObjX, animal_data.EncObjZ, 'o', markerfacecolor=colors_stable[i], markeredgecolor=colors_stable[i], markersize=7)
    
    # Plot locations of unstable animals
    for i, animal in enumerate(unstable_objects):
        # Filter the data for the current object (animal)
        animal_data = periods[periods.EncObj == animal]
        ax.plot(animal_data.EncObjX, animal_data.EncObjZ, 'x', markerfacecolor='white', markeredgecolor=colors_unstable[i], markersize=10)
    
    # Create a legend
    legend_elements = [
        plt.Line2D([0], [0], marker = 'o', markersize = 10, color='grey', lw=0, label='Stable'),
        plt.Line2D([0], [0], marker = 'x', markersize = 10, color='grey', lw=0, label='Unstable'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', frameon=False, ncol=1, fontsize=24, handlelength=1, bbox_to_anchor=(1, 0.7))
    fig.savefig(outputfile, bbox_inches='tight', format = 'svg')
    plt.close()

# Figure 3B: Helping function to add significance bars to figure 3B
def figure3b_add_significance_bars(ax, significant_combinations):
    bottom, top = ax.get_ylim()
    y_range = top - bottom
    
    # Define significance levels
    sig_levels = {0.001: '***', 0.01: '**', 0.05: '*', float('inf'): 'n.s.'}

    for (x1, x2), p in significant_combinations:
        bar_height = 0.82 * top + 0.07 * y_range + 0.12
        bar_tips = bar_height - 0.02 * y_range

        # Plot significance brackets
        ax.plot([x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')

        # Determine significance symbol
        sig_symbol = next(v for k, v in sig_levels.items() if p < k)
        
        # Add the significance symbol
        if sig_symbol == 'n.s.':
            ax.text((x1 + x2) * 0.5, bar_height - 0.001 * y_range, sig_symbol, 
                    ha='center', va='bottom', c='k', fontsize=18)
        else:
            ax.text((x1 + x2) * 0.5, bar_height - 0.015 * y_range, sig_symbol, 
                    ha='center', va='bottom', c='k', fontsize=18)

# Figure 3B: Memory performance as a function of recall type and object stability
def figure3b_influence_object_stability(cohort1, cohort2, cohorts_1and2, significance_combinations, outputfile):
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(6, 7), sharey=True)

    # Define colors
    colors = {
        'Allo_Unstable': 'deepskyblue',
        'Allo_Stable': 'royalblue',
        'Ego_Unstable': 'limegreen',
        'Ego_Stable': 'green'
    }
    
    # Titles for each plot
    titles = ['Cohort 1', 'Cohort 2', 'Cohorts 1 and 2']
    
    # Plot 
    for i, cohort in enumerate([cohort1, cohort2, cohorts_1and2]):
        # Group data by Subject and calculate the performances
        allo_stable_data = cohort[(cohort['StableObj'] == True) & (~cohort['AlloRetRankedPerformance'].isna())].groupby('Subject')['AlloRetRankedPerformance'].mean().values
        allo_unstable_data = cohort[(cohort['StableObj'] == False) & (cohort['AlloRetRankedPerformance'].notna())].groupby('Subject')['AlloRetRankedPerformance'].mean().values
        ego_stable_data = cohort[(cohort['StableObj'] == True) & (cohort['EgoRetRankedPerformance'].notna())].groupby('Subject')['EgoRetRankedPerformance'].mean().values
        ego_unstable_data = cohort[(cohort['StableObj'] == False) & (cohort['EgoRetRankedPerformance'].notna())].groupby('Subject')['EgoRetRankedPerformance'].mean().values


        # Plot boxplot
        data_to_plot = [allo_unstable_data, allo_stable_data, ego_unstable_data, ego_stable_data]
        bplot = axes[i].boxplot(data_to_plot, patch_artist=True, widths=0.9,
                                boxprops=dict(facecolor='lightgray'),  # Default color
                                medianprops=dict(color='black', linewidth=2))  # Customize median color

        # Assign colors to the boxes
        for j in range(len(data_to_plot)):
            bplot['boxes'][j].set_facecolor(colors[list(colors.keys())[j]])  # Set box colors
            bplot['boxes'][j].set_edgecolor('black')  

        # Customize plot
        axes[i].set_xticks([])
        axes[i].set_title(titles[i], fontsize=18)
        axes[i].grid(False)
        axes[i].set_ylim(0.5, 1.05)  # Adjust the y-axis limit as needed

    # Customize plot
    axes[0].set_ylabel('Performance', fontsize=18)
    axes[1].set_yticks([])  
    axes[2].set_yticks([])  
    axes[0].set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    axes[0].set_yticklabels(['0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=18)
    axes[0].tick_params(axis='y', which='both', length=0)
    axes[1].tick_params(axis='y', which='both', length=0)
    axes[2].tick_params(axis='y', which='both', length=0)

    for ax in axes[1:]:
        ax.set_ylabel('')

    # Create a legend
    legend_elements = [
        plt.Line2D([0], [0], color=colors['Allo_Unstable'], lw=2, label='Allocentric unstable'),
        plt.Line2D([0], [0], color=colors['Allo_Stable'], lw=2, label='Allocentric stable'),
        plt.Line2D([0], [0], color=colors['Ego_Unstable'], lw=2, label='Egocentric unstable'),
        plt.Line2D([0], [0], color=colors['Ego_Stable'], lw=2, label='Egocentric stable'),
    ]
    axes[2].legend(handles=legend_elements, loc='lower left', frameon=False, ncol=2, fontsize=16, handlelength=1, bbox_to_anchor=(-2.3, -0.2))

    # Add significance bars
    figure3b_add_significance_bars(axes[0], significance_combinations[0])
    figure3b_add_significance_bars(axes[1], significance_combinations[1])
    figure3b_add_significance_bars(axes[2], significance_combinations[2])

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.1)
    plt.savefig(outputfile, bbox_inches='tight', format = 'svg')
    plt.close()
    
# Figure 4A: Example of object-specific learning curves
def figure4a_weibull_fit(periods_complete, weibull_table, subject, animal, ret_type, ax, is_first_column=False):
    # Extract data for the specified subject and animal
    if ret_type == 'allo':
        y = periods_complete[(periods_complete.Subject == subject) & 
                          (periods_complete.AlloRetObj == animal)].AlloRetRankedPerformance.values
    elif ret_type == 'ego':
        y = periods_complete[(periods_complete.Subject == subject) & 
                          (periods_complete.EgoRetObj == animal)].EgoRetRankedPerformance.values
    x = np.arange(1, len(y) + 1)

    # Get Weibull fit parameters
    df = weibull_table[(weibull_table.subject_id == str(subject)) & 
                              (weibull_table.animal == animal) & 
                              (weibull_table.ret_type == ret_type)]
    k_emp = df.k.values[0]
    lambda_emp = df['lambda'].values[0]
    c_emp = df.c.values[0]
    cp = df.change_point.values[0]
    slope = df.slope.values[0]

    # Generate Weibull values
    y_weibull = LN_Functions.weibull(x, k_emp, lambda_emp, c_emp)

    # Compute the change point value
    y_cp = LN_Functions.weibull(cp, k_emp, lambda_emp, c_emp)

    # Compute tangent lines
    x_before = np.array([cp - 1, cp])  # 1 unit before cp
    y_before = slope * (x_before - cp) + y_cp  # Linear approximation
    x_after = np.array([cp, cp + 1])  # 1 unit after cp
    y_after = slope * (x_after - cp) + y_cp  # Linear approximation

    # Plot actual performance data
    if ret_type == 'allo':
        ax.plot(x, y, 'o', color='blue')  # Allocentric: Blue dots
    elif ret_type == 'ego':
        ax.plot(x, y, 'o', color='green')  # Egocentric: Green dots

    # Plot Weibull fit (dark gray line)
    ax.plot(x, y_weibull, '-', color=(0.3, 0.3, 0.3))  # Dark gray line

    # Add annotations only for the first column
    if is_first_column:
        ax.text(x[len(x)//2], y_weibull[len(x)//2] - 0.13, 'Weibull fit', 
                fontsize=28, color=(0.3, 0.3, 0.3), ha='center')

        # Highlight the change point 
        ax.plot(cp, y_cp, marker='x', color='red', markersize=10, markeredgewidth=3)
        ax.text(cp + 1, y_cp - 0.02, 'Change point with slope', 
                fontsize=28, color='red', ha='left', va='center')

        # Add tangent lines (red dashed lines for slope)
        ax.plot(x_before, y_before, 'r--', lw=3)
        ax.plot(x_after, y_after, 'r--', lw=3)
        ax.text(x[-1] + 0.5, y_weibull[-1] - 0.1, 'Asymptotic \nperformance', 
                fontsize=28, color='black', ha='left', va='bottom')

    # Plot change point and slope in all plots 
    ax.plot(cp, y_cp, marker='x', color='red', markersize=15, markeredgewidth=3)
    ax.plot(x_before, y_before, 'r--', lw=3)
    ax.plot(x_after, y_after, 'r--', lw=3)

    # Customize plots
    ax.set_ylim(0, 1.05)
    ax.set_xticks([5, 10, 15, 20])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
# Figure 4B-E: Histograms from analyses of object-specific learning curves    
def figure4BCDE_hist_learning(allo_stable, allo_unstable, ego_stable, ego_unstable, bin_edges, x_ticks, y_ticks, xlabel, outputfile):
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    
    # Create histograms
    axes[0].hist(allo_stable, bins=bin_edges, color='royalblue', alpha=0.7, edgecolor='black', label='Stable')
    axes[0].hist(allo_unstable, bins=bin_edges, color='deepskyblue', alpha=0.7, edgecolor='black', label='Unstable')
    axes[0].set_title("Allocentric", fontsize = 26)
    axes[0].legend(fontsize =22, frameon = False, loc = 'upper left')
    axes[1].hist(ego_stable, bins=bin_edges, color='green', alpha=0.7, edgecolor='black', label='Stable')
    axes[1].hist(ego_unstable, bins=bin_edges, color='limegreen', alpha=0.7, edgecolor='black', label='Unstable')
    axes[1].set_title("Egocentric", fontsize = 26)
    axes[1].legend(fontsize = 22, frameon = False, loc = 'upper left')
    
    # Customize plots
    for ax in axes:
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='both', labelsize=26)

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Remove the y-axis and y-label for the second plot
    axes[1].yaxis.set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].set_ylabel('')

    # Set the x-label
    fig.text(0.5, 0.05, xlabel, ha='center', fontsize=26)

    # Set the y-label
    axes[0].set_ylabel('Count', fontsize=26, labelpad=10)
    
    # Adjust the plot
    fig.subplots_adjust(right=0.95, bottom = 0.2)
    plt.savefig(outputfile, format = 'svg')
    plt.close()
    
# Figure 6C-F: Associations between allocentric/egocentric direction and memory performance
def figures6cdef_plot_influence_starting_orientation(cohort1, cohort2, cohort3, performance, orientation, y_label, x_label, title1, title2, 
                                        title3, ylim, y_ticks, y_ticklabels, custom_order, cardinal_directions, colors, outputfile):
    
    # Helper function to process data for boxplot
    def process_cohort_data(cohort):
        if orientation == 'AlloStartPosOrient':
            df1 = cohort.groupby(['Subject', 'TrialIdx'])[[orientation]].first().reset_index()
            df2 = cohort.groupby(['Subject', 'TrialIdx'])[[performance]].mean().reset_index()
            df_cohort = pd.merge(df1, df2, on=['Subject', 'TrialIdx'], how='inner')
        else:
            df_cohort = cohort.groupby(['Subject', 'TrialIdx', orientation])[[performance]].mean().reset_index()

        df_cohort = df_cohort.groupby(['Subject', orientation])[[performance]].mean().reset_index()

        # Get data for each orientation to be used in boxplot
        boxplot_data = [df_cohort[df_cohort[orientation] == ori][performance].values for ori in custom_order]

        return boxplot_data

    # Process data for each cohort
    boxplot_data_cohort1 = process_cohort_data(cohort1)
    boxplot_data_cohort2 = process_cohort_data(cohort2)
    boxplot_data_cohort3 = process_cohort_data(cohort3)

    # Create a list of tick labels where only specific cardinal directions are shown
    cardinal_ticklabels = [cardinal_directions.get(i, "") for i in range(len(custom_order))]

    # Create a list of x positions where tick labels are not empty
    cardinal_ticks = [i for i, label in enumerate(cardinal_ticklabels) if label != '']

    # Create figure
    x = np.arange(len(custom_order))  # X positions
    fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)

    plot_data = [
        (boxplot_data_cohort1, title1, colors[0]),
        (boxplot_data_cohort2, title2, colors[1]),
        (boxplot_data_cohort3, title3, colors[2])
    ]

    for i, (ax, (boxplot_data, title, color)) in enumerate(zip(axes, plot_data)):
        # Create the boxplot
        bp = ax.boxplot(boxplot_data, positions=x, patch_artist=True, widths=0.75, 
                        boxprops=dict(facecolor=color), 
                        medianprops=dict(color='black', linewidth=2))

        # Customize plot
        ax.set_title(title, fontsize=28)
        ax.set_ylim(ylim)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticklabels, fontsize=24)
        ax.grid(False)

        # Add cardinal directions as x-tick labels (only for non-empty labels)
        ax.set_xticks(cardinal_ticks)  # Set the x positions where there are cardinal labels
        ax.set_xticklabels([cardinal_ticklabels[i] for i in cardinal_ticks], fontsize=24)  # Set custom tick labels

        # Remove y-ticks for the second and third plots
        if i > 0:
            ax.tick_params(left=False)

    # Add labels and overall title
    axes[0].set_ylabel(y_label, fontsize=26)
    axes[1].set_xlabel(x_label, fontsize=26, labelpad=10)

    # Adjust plot
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15, wspace=0.1)
    plt.savefig(outputfile, format = 'svg')
    plt.close()
    
# Figure 7B: Helping function to add significance bars to figure 7B
def figure7b_add_significance_bars(ax, significant_combinations):
    bottom, top = ax.get_ylim()
    y_range = top - bottom
    
    # Define significance levels
    sig_levels = {0.001: '***', 0.01: '**', 0.05: '*', float('inf'): 'n.s.'}

    for (x1, x2), p in significant_combinations:
        bar_height = 0.82 * top + 0.07 * y_range + 0.08
        bar_tips = bar_height - 0.02 * y_range

        # Plot significance brackets
        ax.plot([x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')

        # Determine significance symbol
        sig_symbol = next(v for k, v in sig_levels.items() if p < k)
        
        # Add the significance symbol
        if sig_symbol == 'n.s.':
            ax.text((x1 + x2) * 0.5, bar_height - 0.001 * y_range, sig_symbol, 
                    ha='center', va='bottom', c='k', fontsize=22)
        else:
            ax.text((x1 + x2) * 0.5, bar_height - 0.05 * y_range, sig_symbol, 
                    ha='center', va='bottom', c='k', fontsize=24)

# Figure 7B: Effects of allocentric/egocentric feedback on allocentric/egocentric memory performance
def figure7b_feedback(periods_complete, significance_combinations, outputfile):
    # Create figure
    plt.figure(figsize=(8, 8))
    x = np.array(['Without', 'With'])
    y_ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    ax_grid = [plt.axes([0.1, 0.6, 0.3, 0.3]), plt.axes([0.6, 0.6, 0.3, 0.3]), 
               plt.axes([0.1, 0.1, 0.3, 0.3]), plt.axes([0.6, 0.1, 0.3, 0.3])]
    
    # Get feedback effects
    ego_without_allofeedback, ego_with_allofeedback, allo_without_egofeedback, allo_with_egofeedback,allo_without_allofeedback, allo_with_allofeedback, ego_without_egofeedback, ego_with_egofeedback = LN_Functions.effect_feedback_on_performance(periods_complete)
    
    data_pairs = [(allo_without_allofeedback, allo_with_allofeedback), 
                  (ego_without_allofeedback, ego_with_allofeedback), 
                  (allo_without_egofeedback, allo_with_egofeedback), 
                  (ego_without_egofeedback, ego_with_egofeedback)]
    
    # Colors for plots
    colors = ['royalblue', '#117711']
    
    for i, ax in enumerate(ax_grid):
        # Customize plot
        ax.set_ylabel('Performance', size=20)
        ax.set_xlabel('Feedback', size=20)
        ax.set_ylim(0.45, 1.1)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, fontsize=20)
        ax.set_xticks([0, 1])  # Set FixedLocator to match x positions
        ax.set_xticklabels(x, fontsize=20)

        # Plot the boxplots 
        data_to_plot = [data_pairs[i][0], data_pairs[i][1]]  # Group data for boxplot
        ax.boxplot(data_to_plot, patch_artist=True, widths=0.8,
                   boxprops=dict(facecolor=colors[i % 2], color='black'),  # Custom colors for boxes
                   capprops=dict(color='black'),  # Whisker cap color
                   whiskerprops=dict(color='black'),  # Whisker color
                   flierprops=dict(marker='o', color='black'),  # Outlier properties
                   medianprops=dict(color='black', linewidth = 2))  # Median line properties

        ax.set_xticks([1,2])
        ax.set_xticklabels(x)

        # Add significance bars
        figure7b_add_significance_bars(ax, significance_combinations[i])

    # Customize plot
    ax0 = plt.axes([0, -0.02, 0.97, 0.99])
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.axis('off')
    ax0.plot([0.5, 0.5], [0, 1], color='black', lw=1)
    ax0.plot([0, 1.05], [0.5, 0.5], color='black', lw=1)

    # Correct the text alignment and rotation
    ax_grid[0].text(1.2, 1.3, 'Allocentric', ha='center', va='center', fontsize=22)
    ax_grid[1].text(1.2, 1.3, 'Egocentric', ha='center', va='center', fontsize=22)
    ax_grid[0].text(-0.8, 0.7, 'Allocentric', ha='center', va='center', rotation=90, fontsize=22)
    ax_grid[2].text(-0.8, 0.7, 'Egocentric', ha='center', va='center', rotation=90, fontsize=22)
    ax_grid[0].text(3, 1.4, 'Performance', ha='center', va='center', fontsize=26)
    ax_grid[2].text(-1.1, 1.3, 'Feedback', ha='center', va='center', rotation=90, fontsize=26)

    plt.savefig(outputfile, bbox_inches='tight', format = 'svg')
    plt.close()

# Figure 7C: Allocentric memory performance as a function of retrieval position
def figure7c_retrieval_position_allo(means_allo_cohort1, means_allo_cohort2, means_allo_combined, outputfile):
    # Labels and colors for the figure
    labels = ['1', '2', '3', '4']
    colors = ['deepskyblue', 'cornflowerblue', 'royalblue']  

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(8, 6), sharey=True)

    # Plot data
    plot_data = [
        (means_allo_cohort1, 'Cohort 1', colors[0]),
        (means_allo_cohort2, 'Cohort 2', colors[1]),
        (means_allo_combined, 'Cohorts 1 and 2', colors[2])
    ]

    for i, (ax, (means, title, color)) in enumerate(zip(axes, plot_data)):
        # Plot boxplot
        ax.boxplot(means, patch_artist=True, widths=0.8,
                   boxprops=dict(facecolor=colors[i], color='black'),  # Custom colors for boxes
                   capprops=dict(color='black'),  # Whisker cap color
                   whiskerprops=dict(color='black'),  # Whisker color
                   flierprops=dict(marker='o', color='black'),  # Outlier properties
                   medianprops=dict(color='black', linewidth = 2))  # Median line properties
        ax.set_title(title, fontsize=26)
        ax.set_ylim(0.45, 1.1)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=24)
        ax.grid(False)
        ax.set_xticklabels(labels, fontsize=24)
        ax.tick_params(left = False, bottom = False)

    # Add labels and overall title
    axes[0].set_ylabel('Allocentric performance', fontsize=26)
    axes[1].set_xlabel('Retrieval position', fontsize=26)

    # Adjust plot
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15, wspace=0.1)
    plt.savefig(outputfile, format = 'svg')
    plt.close()
    
# Figure 7D: Egocentric memory performance as a function of retrieval position    
def figure7d_retrieval_position_allo(means_ego_cohort1, means_ego_cohort2, means_ego_combined, outputfile):
    # Labels and colors for the figure
    labels = ['1', '2', '3', '4']
    colors = ['limegreen', '#1f9b1f', '#117711']  

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(8, 6), sharey=True)

    # Plot data
    plot_data = [
        (means_ego_cohort1, 'Cohort 1', colors[0]),
        (means_ego_cohort2, 'Cohort 2', colors[1]),
        (means_ego_combined, 'Cohorts 1 and 2', colors[2])
    ]

    for i, (ax, (means, title, color)) in enumerate(zip(axes, plot_data)):
        # Plot boxplot
        ax.boxplot(means, patch_artist=True, widths=0.8,
                   boxprops=dict(facecolor=colors[i], color='black'),  # Custom colors for boxes
                   capprops=dict(color='black'),  # Whisker cap color
                   whiskerprops=dict(color='black'),  # Whisker color
                   flierprops=dict(marker='o', color='black'),  # Outlier properties
                   medianprops=dict(color='black', linewidth = 2))  # Median line properties
        ax.set_title(title, fontsize=26)
        ax.set_ylim(0.45, 1.1)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=24)
        ax.grid(False)
        ax.set_xticklabels(labels, fontsize=24)
        ax.tick_params(left = False, bottom = False)

    # Add labels and overall title
    axes[0].set_ylabel('Egocentric performance', fontsize=26)
    axes[1].set_xlabel('Retrieval position', fontsize=26)

    # Adjust plot
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15, wspace=0.1)
    plt.savefig(outputfile, format = 'svg')
    plt.close()

# Figure 8A: Percentage of time gazing at different aspects of the virtual environment
def figure8A_eye_gaze_across_subjects(eye_percentages, outputfile):
    # Collect values
    animals_complete = LN_Functions.collect_values_by_key(eye_percentages)

    # Sort and filter keys
    keys_sorted = ['Objects', 'North fence', 'East fence', 'South fence', 'West fence', 'Trees', 'Ground', 'Flowers', 'Grass', 'Sky', 'Handles'] 
    objects = [key for key in reversed(keys_sorted) if key in animals_complete]
    data = [animals_complete[key] for key in objects]  # Collect lists of values for each object

    # Create figure
    plt.figure(figsize=(10, 8))   
    
    # Plot boxplot
    plt.boxplot(data, vert=False, patch_artist=True, 
                boxprops=dict(facecolor="skyblue", color="black"), 
                medianprops=dict(color="black", linewidth=2),
                widths=0.9)
    plt.xlabel('Time gazing at aspects of the environment (%)', fontsize=34, labelpad=10)
    plt.ylabel('', fontsize=34, labelpad = 10)
    plt.yticks(ticks=range(1, len(objects) + 1), labels=objects, fontsize=30)  # Adjust y-ticks for boxplot
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xticks([0, 10, 20, 30, 40], fontsize=30)
    
    # Adjust plot
    subplot_adj = [0.01, 0.99, 0.9, 0.15, 0.1]
    plt.subplots_adjust(left=subplot_adj[0], right=subplot_adj[1], top=subplot_adj[2], bottom=subplot_adj[3], wspace=subplot_adj[4])
    plt.savefig(outputfile, bbox_inches='tight', format = 'svg')
    plt.close()    
    
# Figure 8B: Definition of the gaze area
def figure8b_explanation_gaze_area(outputfile):
    # Parameters (this is just an example)
    player_start = [0, 0]
    animal_1 = [4, 4]
    animal_2 = [-3, -7]
    diameter = 1

    # Generate random gaze points
    num_points = 10000
    random_gaze_points = np.random.uniform(low=-10, high=10, size=(num_points, 2))

    # Separate points into inside/outside the gaze area
    inside_ellipse_1 = []
    inside_ellipse_2 = []

    # Check whether the gaze points are in the gaze area
    for point in random_gaze_points:
        if LN_Functions.relationship_player_animal(player_start, animal_1, point, diameter):
            inside_ellipse_1.append(point)
        elif LN_Functions.relationship_player_animal(player_start, animal_2, point, diameter):
            inside_ellipse_2.append(point)

    # Transform to numpy arrays
    inside_ellipse_1 = np.array(inside_ellipse_1)
    inside_ellipse_2 = np.array(inside_ellipse_2)

    # Create subplots
    fig, ax = plt.subplots(figsize=(6, 6))

    # Helper function to draw an ellipse
    def draw_ellipse(ax, focus1, focus2, diameter, color, label):
        center_x = (focus1[0] + focus2[0]) / 2
        center_y = (focus1[1] + focus2[1]) / 2
        distance = np.sqrt((focus1[0] - focus2[0])**2 + (focus1[1] - focus2[1])**2)
        angle = math.degrees(math.atan2(focus2[1] - focus1[1], focus2[0] - focus1[0]))
        ellipse = plt.matplotlib.patches.Ellipse(
            (center_x, center_y),
            width=distance,
            height=diameter,
            angle=angle,
            edgecolor=color,
            facecolor='none',
            linestyle='--',
            linewidth=1.5,
            label = label
        )
        ax.add_patch(ellipse)

    # Draw ellipses
    draw_ellipse(ax, player_start, animal_1, diameter, '#008000', label = 'Gaze area for object 1')
    draw_ellipse(ax, player_start, animal_2, diameter, '#0000FF', label = 'Gaze area for object 2')

    # Plot points inside each ellipse
    if len(inside_ellipse_1) > 0:
        ax.scatter(inside_ellipse_1[:, 0], inside_ellipse_1[:, 1], color='#008000', alpha=0.3, s = 3)
    if len(inside_ellipse_2) > 0:
        ax.scatter(inside_ellipse_2[:, 0], inside_ellipse_2[:, 1], color='#0000FF', alpha=0.3, s = 3)

    # Plot player_start and animal positions
    ax.scatter(*player_start, color='red', s=50)
    ax.scatter(*animal_1, color='#008000', s=50)
    ax.scatter(*animal_2, color='#0000FF', s=50)

    # Customize plot
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('x (vu)', fontsize=26, labelpad=-22)
    ax.set_ylabel('z (vu)', rotation = 0, fontsize=26, labelpad = -7)
    ax.set_xticks([-10, 10])
    ax.set_yticks([-10, 10])
    ax.set_xticklabels(['-10', '10'], fontsize = 24)
    ax.set_yticklabels(['-10', '10'], fontsize = 24)
    ax.text(player_start[0] - 6, player_start[1] - 1, 'Starting \nposition', color='red', fontsize=24)
    ax.text(animal_1[0] + 0.5, animal_1[1], 'Object 1', color='#008000', fontsize=24)
    ax.text(animal_2[0] + 0.5, animal_2[1] - 0.5, 'Object 2', color='#0000FF', fontsize=24)
    plt.legend(loc = 'upper left', frameon = False, fontsize = 20, handlelength = 1)
    plt.savefig(outputfile, bbox_inches='tight', format = 'svg') 
    plt.close()
    
    