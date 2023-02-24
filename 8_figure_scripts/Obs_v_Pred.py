import pandas as pd
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from math import pi
import sys
import os
import scipy as sc
import warnings
from numpy import log10, sqrt
from scipy import stats
from PIL import Image
from io import BytesIO

 
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

mydir = os.path.expanduser("~/GitHub/HACRP-HAIs/")

#########################################################################################
################################ FUNCTIONS ##############################################
#########################################################################################

def obs_pred_rsquare(obs, pred):
    # Determines the prop of variability in a data set accounted for by a model
    # In other words, this determines the proportion of variation explained by
    # the 1:1 line in an observed-predicted plot.
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)

def count_pts_within_radius(x, y, radius, logscale=0):
    """Count the number of points within a fixed radius in 2D space"""
    #TODO: see if we can improve performance using KDTree.query_ball_point
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html
    #instead of doing the subset based on the circle
    raw_data = np.array([x, y])
    x = np.array(x)
    y = np.array(y)
    raw_data = raw_data.transpose()
    
    # Get unique data points by adding each pair of points to a set
    unique_points = set()
    for xval, yval in raw_data:
        unique_points.add((xval, yval))
    
    count_data = []
    for a, b in unique_points:
        if logscale == 1:
            num_neighbors = len(x[((log10(x) - log10(a)) ** 2 +
                                   (log10(y) - log10(b)) ** 2) <= log10(radius) ** 2])
        else:        
            num_neighbors = len(x[((x - a) ** 2 + (y - b) ** 2) <= radius ** 2])
        count_data.append((a, b, num_neighbors))
    return count_data



def plot_color_by_pt_dens(x, y, radius, loglog=0, plot_obj=None):
    """Plot bivariate relationships with large n using color for point density

    Inputs:
    x & y -- variables to be plotted
    radius -- the linear distance within which to count points as neighbors
    loglog -- a flag to indicate the use of a loglog plot (loglog = 1)

    The color of each point in the plot is determined by the logarithm (base 10)
    of the number of points that occur with a given radius of the focal point,
    with hotter colors indicating more points. The number of neighboring points
    is determined in linear space regardless of whether a loglog plot is
    presented.
    """
    plot_data = count_pts_within_radius(x, y, radius, loglog)
    sorted_plot_data = np.array(sorted(plot_data, key=lambda point: point[2]))

    if plot_obj == None:
        plot_obj = plt.axes()
        
    plot_obj.scatter(sorted_plot_data[:, 0],
            sorted_plot_data[:, 1],
            facecolors='none',
            s = 30, edgecolors='0.1', linewidths=0.75, #cmap='Greys_r',
            )
    # plot points
    c = np.array(sorted_plot_data[:, 2])**0.25
    c = np.max(c) - c
    plot_obj.scatter(sorted_plot_data[:, 0],
                    sorted_plot_data[:, 1],
                    c = c,
                    s = 30, edgecolors='k', linewidths=0.0, #cmap='Greys_r',
                    #alpha = 0.5,
                    )
        
    return plot_obj





fdates = ['2021_04']

for fdate in fdates:
    #########################################################################################
    ########################## IMPORT HAI DATA ##############################################
    #########################################################################################

    CAUTI_df = pd.read_pickle(mydir + "2_optimize_random_sampling_models/optimized_by_HAI_file_date/CAUTI/CAUTI_Data_opt_for_SIRs_" + fdate + ".pkl")
    print('CAUTI_df.shape:', CAUTI_df.shape)
    CAUTI_df = CAUTI_df[CAUTI_df['CAUTI Predicted Cases'] >= 1]
    CAUTI_df = CAUTI_df[CAUTI_df['CAUTI Urinary Catheter Days'] > 0]
    CAUTI_df = CAUTI_df[CAUTI_df['simulated O/E'] >= 0]
    CAUTI_df = CAUTI_df[CAUTI_df['O/E'] >= 0]
    print('CAUTI_df.shape:', CAUTI_df.shape)
    
    CLABSI_df = pd.read_pickle(mydir + "2_optimize_random_sampling_models/optimized_by_HAI_file_date/CLABSI/CLABSI_Data_opt_for_SIRs_" + fdate + ".pkl")
    print('CLABSI_df.shape:', CLABSI_df.shape)
    CLABSI_df = CLABSI_df[CLABSI_df['CLABSI Predicted Cases'] >= 1]
    CLABSI_df = CLABSI_df[CLABSI_df['CLABSI Number of Device Days'] > 0]
    CLABSI_df = CLABSI_df[CLABSI_df['simulated O/E'] >= 0]
    CLABSI_df = CLABSI_df[CLABSI_df['O/E'] >= 0]
    print('CLABSI_df.shape:', CLABSI_df.shape)
    
    MRSA_df = pd.read_pickle(mydir + "2_optimize_random_sampling_models/optimized_by_HAI_file_date/MRSA/MRSA_Data_opt_for_SIRs_" + fdate + ".pkl")
    print('MRSA_df.shape:', MRSA_df.shape)
    MRSA_df = MRSA_df[MRSA_df['MRSA Predicted Cases'] >= 1]
    MRSA_df = MRSA_df[MRSA_df['MRSA patient days'] > 0]
    MRSA_df = MRSA_df[MRSA_df['simulated O/E'] >= 0]
    MRSA_df = MRSA_df[MRSA_df['O/E'] >= 0]
    print('MRSA_df.shape:', MRSA_df.shape)
    
    CDIFF_df = pd.read_pickle(mydir + "2_optimize_random_sampling_models/optimized_by_HAI_file_date/CDIFF/CDIFF_Data_opt_for_SIRs_" + fdate + ".pkl")
    print('CDIFF_df.shape:', CDIFF_df.shape)
    CDIFF_df = CDIFF_df[CDIFF_df['CDIFF Predicted Cases'] >= 1]
    CDIFF_df = CDIFF_df[CDIFF_df['CDIFF patient days'] > 0]
    CDIFF_df = CDIFF_df[CDIFF_df['simulated O/E'] >= 0]
    CDIFF_df = CDIFF_df[CDIFF_df['O/E'] >= 0]
    print('CDIFF_df.shape:', CDIFF_df.shape)
    
    l1 = list(set(CDIFF_df['Facility ID'].tolist()))
    l2 = list(set(CAUTI_df['Facility ID'].tolist()))
    l3 = list(set(CLABSI_df['Facility ID'].tolist()))
    l4 = list(set(MRSA_df['Facility ID'].tolist()))
    
    hosps = list(set(l1 + l2 + l3 + l4))
    print(len(hosps))
    
    '''
    SSI_AH_df = pd.read_pickle(mydir + "2_optimize_random_sampling_models/optimized_by_HAI_file_date/SSI-AH/SSI-AH_Data_opt_for_SIRs_" + fdate + ".pkl")
    SSI_AH_df = SSI_AH_df[SSI_AH_df['SSI: Abdominal Predicted Cases'] >= 1]
    SSI_AH_df = SSI_AH_df[SSI_AH_df['SSI: Abdominal, Number of Procedures'] > 0]
    SSI_AH_df = SSI_AH_df[SSI_AH_df['simulated O/E'] >= 0]
    SSI_AH_df = SSI_AH_df[SSI_AH_df['O/E'] >= 0]
    
    SSI_CP_df = pd.read_pickle(mydir + "2_optimize_random_sampling_models/optimized_by_HAI_file_date/SSI-CP/SSI-CP_Data_opt_for_SIRs_" + fdate + ".pkl")
    SSI_CP_df = SSI_CP_df[SSI_CP_df['SSI: Colon Predicted Cases'] >= 1]
    SSI_CP_df = SSI_CP_df[SSI_CP_df['SSI: Colon, Number of Procedures'] > 0]
    SSI_CP_df = SSI_CP_df[SSI_CP_df['simulated O/E'] >= 0]
    SSI_CP_df = SSI_CP_df[SSI_CP_df['O/E'] >= 0]
    '''
    
    #########################################################################################
    ##################### DECLARE FIGURE 3A OBJECT ##########################################
    #########################################################################################

    fig = plt.figure(figsize=(11, 11))
    rows, cols = 3, 3
    fs = 14
    radius = 1

    matplotlib.rcParams['legend.handlelength'] = 0
    matplotlib.rcParams['legend.numpoints'] = 1

    #########################################################################################
    ################################ GENERATE FIGURE ########################################
    #########################################################################################

    ################################## SUBPLOT 1 ############################################

    ax1 = plt.subplot2grid((rows, cols), (0, 0), colspan=1, rowspan=1)
    #ax1.set_xticks([0, 2, 4, 6, 8, 10, 12])

    x = CAUTI_df['expected O']**0.5
    y = CAUTI_df['CAUTI Observed Cases']**0.5
    r2 = obs_pred_rsquare(y, x)
    print('CAUTI r2:', np.round(r2,3))
    
    slope, intercept, r, p, se = sc.stats.linregress(x, y)
    maxv = min([np.max(x), np.max(y)])
    plt.plot([0, maxv], [0, maxv], 'k', linewidth=2)
    
    plot_color_by_pt_dens(x, y, radius, plot_obj=ax1)
    plt.tick_params(axis='both', labelsize=fs-4)
    plt.xlabel(r'$\sqrt{Random\ expectation}$', fontsize=fs)
    plt.ylabel(r'$\sqrt{CAUTI\ cases}$', fontsize=fs)
    plt.text(0.0*max(x), 0.94*max(y), r'$r^{2}$' + ' = ' + str(np.round(r2, 2)), fontsize=fs)
    
    ################################## SUBPLOT 2 ############################################

    ax2 = plt.subplot2grid((rows, cols), (0, 1), colspan=1, rowspan=1)
    x = CLABSI_df['expected O']**0.5
    y = CLABSI_df['CLABSI Observed Cases']**0.5
    r2 = obs_pred_rsquare(y, x)
    print('CLABSI r2:', np.round(r2,3))
    
    maxv = min([np.max(x), np.max(y)])
    plt.plot([0, maxv], [0, maxv], 'k', linewidth=2)
    
    plot_color_by_pt_dens(x, y, radius, plot_obj=ax2)
    plt.tick_params(axis='both', labelsize=fs-4)
    plt.xlabel(r'$\sqrt{Random\ expectation}$', fontsize=fs)
    plt.ylabel(r'$\sqrt{CLABSI\ cases}$', fontsize=fs)
    plt.text(0.0*max(x), 0.94*max(y), r'$r^{2}$' + ' = ' + str(np.round(r2, 2)), fontsize=fs)
    
    
    ################################## SUBPLOT 3 ############################################

    ax3 = plt.subplot2grid((rows, cols), (1, 0), colspan=1, rowspan=1)

    x = MRSA_df['expected O']**0.5
    y = MRSA_df['MRSA Observed Cases']**0.5
    r2 = obs_pred_rsquare(y, x)
    print('MRSA r2:', np.round(r2,3))
    
    slope, intercept, r, p, se = sc.stats.linregress(x, y)
    maxv = min([np.max(x), np.max(y)])
    plt.plot([0, maxv], [0, maxv], 'k', linewidth=2)
    
    plot_color_by_pt_dens(x, y, radius, plot_obj=ax3)
    plt.tick_params(axis='both', labelsize=fs-4)
    plt.xlabel(r'$\sqrt{Random\ expectation}$', fontsize=fs)
    plt.ylabel(r'$\sqrt{MRSA\ cases}$', fontsize=fs)
    plt.text(0.0*max(x), 0.94*max(y), r'$r^{2}$' + ' = ' + str(np.round(r2, 2)), fontsize=fs)
    
    
    ################################## SUBPLOT 4 ############################################

    ax4 = plt.subplot2grid((rows, cols), (1, 1), colspan=1, rowspan=1)

    x = CDIFF_df['expected O']**0.5
    y = CDIFF_df['CDIFF Observed Cases']**0.5
    r2 = obs_pred_rsquare(y, x)
    print('CDIFF r2:', np.round(r2,3))
    
    slope, intercept, r, p, se = sc.stats.linregress(x, y)
    maxv = min([np.max(x), np.max(y)])
    plt.plot([0, maxv], [0, maxv], 'k', linewidth=2)
    
    plot_color_by_pt_dens(x, y, radius, plot_obj=ax4)
    plt.tick_params(axis='both', labelsize=fs-4)
    plt.xlabel(r'$\sqrt{Random\ expectation}$', fontsize=fs)
    plt.ylabel(r'$\sqrt{CDIFF\ cases}$', fontsize=fs)
    plt.text(0.0*max(x), 0.94*max(y), r'$r^{2}$' + ' = ' + str(np.round(r2, 2)), fontsize=fs)
    
    '''
    ################################## SUBPLOT 5 ############################################

    ax5 = plt.subplot2grid((rows, cols), (1, 1), colspan=1, rowspan=1)

    x = SSI_AH_df['expected O']**0.5
    y = SSI_AH_df['SSI: Abdominal Observed Cases']**0.5
    r2 = obs_pred_rsquare(y, x)
    
    slope, intercept, r, p, se = sc.stats.linregress(x, y)
    maxv = min([np.max(x), np.max(y)])
    plt.plot([0, maxv], [0, maxv], 'k', linewidth=2)
    
    plot_color_by_pt_dens(x, y, radius, plot_obj=ax5)
    plt.tick_params(axis='both', labelsize=fs-4)
    plt.xlabel(r'$\sqrt{Random\ expectation}$', fontsize=fs)
    plt.ylabel(r'$\sqrt{SSI-AH\ cases}$', fontsize=fs)
    plt.text(0.0*max(x), 0.94*max(y), r'$r^{2}$' + ' = ' + str(np.round(r2, 2)), fontsize=fs)
    
    ################################## SUBPLOT 6 ############################################

    ax6 = plt.subplot2grid((rows, cols), (1, 2), colspan=1, rowspan=1)

    x = SSI_CP_df['expected O']**0.5
    y = SSI_CP_df['SSI: Colon Observed Cases']**0.5
    r2 = obs_pred_rsquare(y, x)
    
    slope, intercept, r, p, se = sc.stats.linregress(x, y)
    maxv = min([np.max(x), np.max(y)])
    plt.plot([0, maxv], [0, maxv], 'k', linewidth=2)
    
    plot_color_by_pt_dens(x, y, radius, plot_obj=ax6)
    plt.tick_params(axis='both', labelsize=fs-4)
    plt.xlabel(r'$\sqrt{Random\ expectation}$', fontsize=fs)
    plt.ylabel(r'$\sqrt{SSI-CP\ cases}$', fontsize=fs)
    plt.text(0.0*max(x), 0.94*max(y), r'$r^{2}$' + ' = ' + str(np.round(r2, 2)), fontsize=fs)
    '''
    
    #########################################################################################
    ################################ FINAL FORMATTING #######################################
    #########################################################################################
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(mydir+'/figures/Obs_v_Pred.png', dpi=400, bbox_inches = "tight")



