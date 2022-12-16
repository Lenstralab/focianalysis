import os
import sys
import re
import shutil
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib.pyplot as plt
from time import time
from tllab_common.wimread import imread as imr
from tllab_common import misc
from warnings import simplefilter
from math import log10, floor

if __package__ is None or __package__ == '':
    import _version
else:
    from . import _version

simplefilter(action='ignore', category=pd.errors.PerformanceWarning)  # TODO: fix instead of ignore


# --- Set sizes of plot text ---
scalar = 2.5
font_size = 11
plt.rc('font', size=font_size)            # controls default text sizes
plt.rc('axes', titlesize=font_size)       # fontsize of the axes title
plt.rc('axes', labelsize=font_size)       # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_size)      # fontsize of the tick labels
plt.rc('ytick', labelsize=font_size)      # fontsize of the tick labels
plt.rc('legend', fontsize=font_size)      # legend fontsize
plt.rc('figure', titlesize=font_size)     # fontsize of the figure title
plt.rcParams['figure.figsize'] = (scalar, scalar)    # size of plots
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['pdf.fonttype'] = 42

# Optional
# plt.style.use('dark_background')
plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams["scatter.edgecolors"] = 'none'
plt.rcParams["scatter.edgecolors"] = 'none'
plt.rcParams["figure.facecolor"] = 'none'
plt.rcParams["axes.facecolor"] = 'none'
ecolor = 'grey'
alpha = 0.5


def get_rep_paths(params):
    """
    Create an ordered dictionary with for each condition (= key) a list of lists with [path_in, rep_name, cell_nr]
    """
    ld_rep_paths = {}  # dictionary is ordered to remember order of keys (= conditions) added.
    if not params['conditions_to_analyze']:
        raise FileNotFoundError(f"No conditions specified")
    for condition in params['conditions_to_analyze']:  # loop over conditions
        ld_rep_paths[condition] = []

        # new data
        for path_in in params['path_in']:
            date = re.search(r'\d{8}', path_in)[0]
            path_in_check = os.path.join(path_in, f"{date}_loc_results_{params['color']}_")
            rep_nr = 0
            rep_name = condition.replace(' ', '_') + '_'

            # add replicate data to values of condition
            while os.path.exists(file := f'{path_in_check}{rep_name}{rep_nr}.txt'):
                cell_nr = len(pd.read_csv(file, sep='\t', usecols=['file', 'cell']).query("cell>0").groupby(['file', 'cell']))
                if cell_nr >= 0:  # only add data if cells are found
                    ld_rep_paths[condition].append([os.path.join(path_in, date), f'{rep_name}{rep_nr}', cell_nr])
                    rep_nr += 1

    # remove conditions for which no data was found
    ld_rep_paths = {key: values for key, values in ld_rep_paths.items() if len(values) > 0}

    # check if data was found
    if len(ld_rep_paths.keys()) == 0:
        raise FileNotFoundError(f"No data found... check input folders, conditions and color.")
    if len(ld_rep_paths.keys()) != len(params['conditions_to_analyze']):
        print(f"No data was found for the following conditions:")
        for con in params['conditions_to_analyze']:
            if con not in ld_rep_paths.keys():
                print(f"{con}")
    return ld_rep_paths


def plot_name(params, ld_rep_paths):
    """
    Make plot name and date
    """
    if params['plotCom'] == 0 or isinstance(params['plotCom'], str) and params['plotCom'].lower().startswith('rep'):
        params['plotCom'] = 0
        print('Making plots for replicates')
        name_plot = [key.replace(' ', '_') for key in ld_rep_paths]  # list of condition names for which data was found
    elif params['plotCom'] == 1 or isinstance(params['plotCom'], str) and params['plotCom'].lower().startswith('com'):
        params['plotCom'] = 1
        print('Making plots for conditions')
        name_plot = ['_'.join(ld_rep_paths).replace(' ', '_')]  # list with joined condition names as only item
        if len(name_plot[0]) > 125:  # shorten file name if too long
            name_plot = [f"{next(iter(ld_rep_paths)).replace(' ', '_')}_{next(reversed(ld_rep_paths)).replace(' ', '_')}"]
    else:
        raise ValueError('Please define plotCom as 0 (replicate plots) or 1 (combined plots)')
    return name_plot


def plot_num(params, ld_rep_paths, name_plot):
    """
    Make plots of foci number per cell
    """
    print('plotNum')
    x_max = params['plotNum']['x_max']  # maximum number of foci per cell used to determine the width of the last histogram bin
    y_max = params['plotNum']['y_max']  # maximum frequency height on y-axis
    location = params['location']  # cellular location of foci ("cell" or "nucleus")

    # plot text
    legend_com = []  # list of legendNames for each condition including number of replicates
    legend_sep = []  # list with for each condition a list with legendNames from each replicate
    dates_all = []  # list of dates from each replicate within a condition
    data_txt = '\n\nCount Data:\n'  # string containing all counts

    # plot data
    count_all = []  # list of lists containing foci numbers per replicate per condition
    x_max_data = x_max # maximum number of foci per cell used to determine the width of the last histogram bin

    for condition, replicates in ld_rep_paths.items():
        # plot text
        rep_names = ''  # list with replicate names
        rep_dates = []  # list with dates (folders) from which data was used
        cell_nr_com = 0  # number of cells in condition
        data_txt += f'\n{condition}\n'

        # plot data
        count_con = []  # list of number of foci per cell

        for rep_ID in replicates:
            path_in, rep_name, cell_nr = rep_ID
            rep_names = f"{rep_names} {path_in.split('/')[-1]} {rep_name.replace('_', ' ')} ({cell_nr} cells)\n"
            rep_dates.append(path_in.split('/')[-1])
            cell_nr_com += cell_nr

            try:
                # load data and filter out spots which are not inside a cell or have a R2_peak below -1
                df_count = pd.read_csv(f"{path_in}_loc_results_{params['color']}_{rep_name}.txt", sep='\t',
                                       # usecols=['file', location, 'fit']).query(f"{location}>0")
                                       usecols=['file', location, 'R2_peak', 'fit']).query(f"{location}>0 & R2_peak>-1|R2_peak!=R2_peak")
                # group foci (represented by fit (True/False for foci, NaN for empty cell)) and get counts per cell
                count_rep = df_count.groupby(['file', location])['fit'].apply(lambda x: (x==True).sum()).to_list()
            except:
                count_rep = np.loadtxt(f"{path_in}_CountBoth_{params['color']}_{rep_name}.txt")

            data_txt += f"{path_in.split('/')[-1]} {rep_name.replace('_', ' ')} ({cell_nr} cells)\n{str(count_rep)}\n"
            count_con.append(count_rep)
            if max(count_rep) > x_max_data:  # if the maximum abundancy exceeds x_max, replace it
                x_max_data = int(max(count_rep))

        # plot text
        rep_nr = len(ld_rep_paths[condition])
        legend_com.append(f'{condition} (n = {rep_nr}, {cell_nr_com} cells)')
        legend_sep.append(condition + '\n' + rep_names[:-1])

        # plot data
        dates_all.append(np.unique(rep_dates).tolist())
        count_all.append(count_con)

    # plot date (first part of plot name)
    dates_plot = np.unique(np.concatenate(dates_all)).tolist()
    dates_plot_name = dates_plot[0] if len(dates_plot) == 1 else f"{dates_plot[0]}-{dates_plot[-1]}"

    # create path_out containing dates of first and last folder in which replicates were found
    if params['plotCom'] == 1:
        type_plot = 'Com'  # combined replicates
        legend_plot = legend_com
    else:
        type_plot = 'Sep'  # separate replicates
        legend_plot = legend_sep

    # create path_out containing dates of first and last folder in which replicates were found
    path_out = os.path.join(params['path_out'], f"{dates_plot_name}_Num{type_plot}_{params['location']}_{params['color']}_{name_plot[0]}")

    # compute the Kruskal-Wallis H-test for independent samples and put result in plot test
    con_nrs = len(count_all)
    count_all_com = [np.hstack(count_all[rep]).tolist() for rep in range(con_nrs)]
    if con_nrs > 1:
        statistic, pvalue = ss.kruskal(*count_all_com)
        if pvalue <= 0.05:
            text = 'population median of the groups are unequal \n (Kruskal-Wallis = %.3g; p = %.3g)\n' \
                   % (statistic, pvalue)
            for item in legend_com:
                text += f"{legend_com.index(item)+1}) {item}\n"
                print(f"{legend_com.index(item)+1}) {item}")
            text += f"\nMann-Whitney U test (two-sided, without p-value correction)\n" \
                    f"{sp.posthoc_mannwhitney(count_all_com, alternative='two-sided', p_adjust = None)}"
            print(f"\nMann-Whitney U test (two-sided, without p-value correction)\n"
                  f"{sp.posthoc_mannwhitney(count_all_com, alternative='two-sided', p_adjust = None)}")
        else:
            text = 'Population median of the groups are equal \n (Kruskal-Wallis U = %.3g; p = %.3g)\n' \
                   % (statistic, pvalue)
    else:
        text = None

    # set binrange for x-axis
    binrange = list(range(x_max))
    binrange.append(x_max_data)  # set right edge of last bin to include maximum count (so unequal bin size)

    # set upper limit of y-axis
    for i in range(len(count_all)):
        for j in range(len(count_all[i])):
            rep_hist = np.histogram(count_all[i][j], bins=binrange)[0] / len(count_all[i][j])  # normalized histogram
            rep_hist_max = np.max(rep_hist)
            if rep_hist_max > y_max:
                rep_hist_max = rep_hist_max + 0.05  # Add half of 0.1 to round up to next .1 decimal
                y_max = rep_hist_max.round(1)
    if params['plotCom'] == 1:
        y_max = params['plotNum']['y_max']

    # calculate step size of yticks for frequency plots using a fixed tick number and 1 decimal
    y_step = 0.1
    y_tick_nr = 3
    while y_max/y_tick_nr > y_step:
        y_step += 0.1
    height = con_nrs*scalar/2

    # plot histograms of foci abundancy
    fig, axs = plt.subplots(figsize=(scalar*0.8, height), ncols=1, nrows=con_nrs, sharex=True)

    # loop over conditions for which data was found
    for i, ax in enumerate(fig.axes):
        rep_nrs = len(count_all[i])  # number of replicates of current condition
        rep_hist = np.array([(np.histogram(count_all[i][j], bins=binrange)[0] / len(count_all[i][j])).tolist()
                             for j in range(rep_nrs)])  # list of normalized replicate histograms
        rep_hist_concat = np.concatenate(np.dstack(rep_hist)[0])   # concatenated list of counts histograms
        con_data = np.concatenate(count_all[i])
        
        n_data = [len(rep) for rep in count_all[i]]  # number of data points per rep
        n_cells = [len(rep) for rep in count_all[i]]  # number of cells per rep
        con_weights = weights(n_data, n_cells)
        con_hist, con_sem = bootstrap(con_data, con_weights, binrange)

        if params['plotCom'] == 1:
            # plot histograms of combined foci abundancy frequencies
            ax.bar(np.arange(x_max) - 0.5, con_hist, width=0.9,
                   label=legend_plot[i], align='edge', color=params['color_palette'][i], yerr=con_sem,
                   error_kw=dict(ecolor=ecolor), capsize=2)
        elif params['plotCom'] == 0:
            # plot histograms of replicate foci abundancy frequencies
            xcoordinates = []
            stepsize = 0.9/rep_nrs
            for j in range(x_max):
                for l in range(rep_nrs):
                    xcoordinates.append(j+l*stepsize)
            ax.bar(np.array(xcoordinates) - 0.5, rep_hist_concat, width=0.9/rep_nrs, label=legend_plot[i], align='edge', 
                   color=params['color_palette'][i])
        ax.set_xticks(np.arange(x_max))
        ax.set_xlim(-0.6, x_max-0.5)
        ax.set_yticks(np.arange(0, x_max, step=y_step))
        ax.set_ylim(0, y_max)
        if i == con_nrs-1:
            ax.set_xticklabels(np.arange(x_max-1).tolist() + [r' $\geq$'+ str(x_max-1)])
        if params['legend'] == 1:
            ax.legend(bbox_to_anchor=(1, 0.5), loc='center left')
    plt.xlabel(f"{params['foci_name']} count per cell")
    fig.supylabel("frequency", x=-0.12)
    plt.savefig(f"{path_out}{params['file_type']}", bbox_inches='tight')
    plt.close(fig)

    # create metadata file with information about pipeline version and replicates used
    with open(os.path.abspath(_version.__file__)) as f:
        txt_out = f.read()
    txt_out += '\nReplicates analyzed:\n' + '\n'.join(np.hstack(legend_sep))
    if bool(text):
        txt_out += '\n\n' + text
    # add count data
    txt_out += data_txt
    np.savetxt(f"{path_out}.txt", [txt_out], fmt='%s')


def get_y_max(num):
    """
    Creates a sensible y_max value out of the maximum data point (i.e. 1 -> 1, 1.01 -> 1.1, 835436 -> 840000)
    """
    return np.ceil(num/10**floor(log10(num))*10)*10**floor(log10(num)-1)


def plot_par(params, ld_rep_paths, name_plot):
    """ Make plots of foci params """
    print('plotPar')

    ch_foci = params['color']
    location = params['location']
    plt_params = params['plt_params']
    cols = np.unique(['fit', 'file', 'R2_peak', location] + plt_params)

    # plot text
    legend_com = []  # list of legendNames for each condition including number of replicates
    legend_sep = []  # list with for each condition a list with legendNames from each replicate
    dates_all = []  # list of dates from each replicate within a condition

    # plot data
    df_all = pd.DataFrame()

    for condition, replicates in ld_rep_paths.items():
        rep_names = []  # list with replicate names
        rep_dates = []  # list with dates (folders) from which data was used
        cell_nr_com = 0  # number of cells in condition
        spot_nr_com = 0 # number of spots in condition

        for rep_ID in replicates:
            path_in, rep_name, cell_nr = rep_ID

            # plot text
            rep_names.append(f"{path_in.split('/')[-1]} {rep_name.replace('_', ' ')} ({cell_nr} cells)")
            rep_dates.append(path_in.split('/')[-1])
            cell_nr_com += cell_nr

            # load foci in subcellular location which are fit
            loc_rep = pd.read_csv(f"{path_in}_loc_results_{ch_foci}_{rep_name}.txt", sep='\t', usecols=cols).query(
                                  f"fit==True & R2_peak>-1 & {location}>0").drop(columns=['fit']).set_index(['file', 'cell'])
            loc_rep['replicate'] = f"{path_in.split('/')[-1]}_{rep_name}"
            loc_rep['condition'] = condition
            spot_nr_com += len(loc_rep)

            # add loc_rep dataframe to loc_com dataframe
            df_all = pd.concat((df_all, loc_rep), ignore_index=True)

        rep_nr = len(replicates)  # number of replicates in condition

        legend_com.append(f'{condition} (n = {rep_nr}, {cell_nr_com} cells, {spot_nr_com} spots)')
        legend_sep.append(rep_names)
        dates_all.append([x for i, x in enumerate(rep_dates) if rep_dates.index(x) == i])

    # set df_plot indexes to condition in order to select replicates from same condition to make plots later on
    df_all = df_all.set_index(['condition', 'replicate'])

    # calculate sigma xy in um
    if 's' in plt_params:
        print('calculating sigma xy in μm')
        df_all['s_um'] = df_all['s'] * params['pixel_size_xy']
        plt_params = list(map(lambda x: x.replace('s', 's_um'),plt_params))
    if 'sz' in plt_params:
        print('calculating sigma z in μm')
        df_all['sz_um'] = df_all['sz'] * params['pixel_size_z']
        plt_params = list(map(lambda x: x.replace('sz', 'sz_um'),plt_params))

    # set plot labels/legends
    color_plot = params['color_palette']

    if params['plotCom'] == 0:
        type_plot = 'Sep'
        sort_plot = 'replicate'
        legend_plot = legend_sep
        dates_plot = dates_all
    elif params['plotCom'] == 1:
        type_plot = 'Com'
        sort_plot = 'condition'
        legend_plot = [legend_com]
        dates_plot = [[x for i, x in enumerate(dates_all[0]) if dates_all[0].index(x) == i]]

    # make plots
    for condition, legend_plot, date_plot, name_plot in zip(ld_rep_paths, legend_plot, dates_plot, name_plot):
        # create path_out containing dates of first and last folder in which replicates were found
        path_out = os.path.join(params['path_out'], date_plot[0])
        if len(date_plot) > 1:
            path_out = f'{path_out}-{date_plot[-1]}'
        path_out = f"{path_out}_Par{type_plot}_{location}_{ch_foci}"

        # plot data
        if params['plotCom'] == 0:
            # select only parts of data corresponding to the condition ld_rep_paths.keys()[i] to make plots
            df_plot = df_all.loc[condition]
        elif params['plotCom'] == 1:
            # select all data to make plots
            df_plot = df_all

        df_plot.to_csv(f"{path_out}_df_plot_{name_plot}.txt", sep='\t')  # save plot df
        sort_values = df_plot.index.unique(level=sort_plot).tolist()  # unique replicates/conditions in order of conditions_to_analyze
        con_nrs = len(sort_values)
        df_plot.sort_index(inplace=True)  # sort df_plot to speed up df_plot.loc[]

        # --- Parameter plots ---
        for i in range(len(plt_params)):
            par = plt_params[i]
            print(f'plotPar {par}')

            label = params['plt_params_labels'][i]
            y_min = params['plt_params_y_min'][i]
            y_max = params['plt_params_y_max'][i]

            if not y_max:  # calculate best y_max value if not specified
                par_max = np.max(df_plot[par])
                y_max = get_y_max(par_max)

            # compute the Kruskal-Wallis H-test for independent samples and put result in plot test
            par_values_all = [df_plot.loc[[sv]][par].to_list() for sv in sort_values]
            text = None
            title = None
            if con_nrs > 1:
                statistic, pvalue = ss.kruskal(*par_values_all)
                if pvalue <= 0.05:
                    text = 'population median of the groups are unequal \n (Kruskal-Wallis = %.3g; p = %.3g)\n' \
                           % (statistic, pvalue)
                    title = text
                    for item in legend_com:
                        text += f"{legend_com.index(item)+1}) {item}\n"
                        print(f"{legend_com.index(item)+1}) {item}")
                    text += f"\nMann-Whitney U test (two-sided, without p-value correction)\n" \
                            f"{sp.posthoc_mannwhitney(par_values_all, alternative='two-sided', p_adjust = None)}"
                else:
                    text = 'Population median of the groups are equal \n (Kruskal-Wallis U = %.3g; p = %.3g)\n' \
                           % (statistic, pvalue)
                    title = text
                print(text)

            # Make jitterplot
            fig, axs = plt.subplots(figsize=(scalar/2, scalar))
            sns.stripplot(data=par_values_all, palette=color_plot, alpha=alpha)
            if params['legend'] == 1:
                leg = plt.legend(legend_plot, bbox_to_anchor=(1, 0.5), loc='center left',
                                 title=title, markerscale=2)
                for lh in leg.legendHandles:
                    lh.set_alpha(1)
            sns.boxplot(data=par_values_all, color='lightgrey', showfliers=False)
            plt.ylabel(label)
            plt.ylim(y_min, y_max)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(-2, 0))
            axs.yaxis.get_offset_text().set_visible(params['plt_sci_text'])
            plt.xticks(np.arange(len(legend_plot)), len(legend_plot) * [])
            plt.savefig(f"{path_out}_{par}_{name_plot}{params['file_type']}", bbox_inches='tight')
            plt.close(fig)

            # create metadata file with information about pipeline version and replicates used
            with open(os.path.abspath(_version.__file__)) as f:
                txt_out = f.read()
            txt_out += '\nReplicates analyzed:\n' + '\n'.join(np.hstack(legend_sep))
            if bool(text):
                txt_out += '\n\n' + text
            np.savetxt(f"{path_out}_{par}_{name_plot}.txt", [txt_out], fmt='%s')


def plot_nr_par(params, ld_rep_paths, name_plot):
    """ Make plots of foci r vs params """
    print('plotNrPar')

    ch_foci = params['color']
    location = params['location']
    plt_params = params['plt_params']
    cols = np.unique(['fit', 'file', 'R2_peak', location] + plt_params)

    # plot text
    legend_com = []  # list of legendNames for each condition including number of replicates
    legend_sep = []  # list with for each condition a list with legendNames from each replicate
    dates_all = []  # list of dates from each replicate within a condition

    # plot data
    df_all = pd.DataFrame()

    for condition, replicates in ld_rep_paths.items():
        rep_names = []  # list with replicate names
        rep_dates = []  # list with dates (folders) from which data was used
        cell_nr_com = 0  # number of cells in condition

        for rep_ID in replicates:
            path_in, rep_name, cell_nr = rep_ID

            # plot text
            rep_names.append(f"{path_in.split('/')[-1]} {rep_name.replace('_', ' ')} ({cell_nr} cells)")
            rep_dates.append(path_in.split('/')[-1])
            cell_nr_com += cell_nr

            # load foci in subcellular location which are fit
            loc_rep = pd.read_csv(f"{path_in}_loc_results_{ch_foci}_{rep_name}.txt", sep='\t',
                                  usecols=cols).query(f"fit==True & R2_peak>-1 & {location}>0").drop(columns=['fit'])
            loc_rep['Nr'] = loc_rep.groupby(['file', location])[location].transform('count')
            loc_rep['replicate'] = f"{path_in.split('/')[-1]}_{rep_name}"
            loc_rep['condition'] = condition
            loc_rep.set_index(['file', location])

            # add loc_rep dataframe to loc_com dataframe
            df_all = pd.concat((df_all, loc_rep), ignore_index=True)

        rep_nr = len(replicates)  # number of replicates in condition
        legend_com.append(f'{condition} ({cell_nr_com} cells from {rep_nr} replicates)')
        legend_sep.append(rep_names)
        dates_all.append([x for i, x in enumerate(rep_dates) if rep_dates.index(x) == i])

    # set df_plot indexes to condition in order to select replicates from same condition to make plots later on
    df_all = df_all.set_index(['condition', 'replicate'])

    # set plot labels/legends
    color_plot = params['color_palette']

    if params['plotCom'] == 0:
        type_plot = 'Sep'
        sort_plot = 'replicate'
        legend_plot = legend_sep
        dates_plot = dates_all
    elif params['plotCom'] == 1:
        type_plot = 'Com'
        sort_plot = 'condition'
        legend_plot = [legend_com]
        dates_plot = [[x for i, x in enumerate(dates_all[0]) if dates_all[0].index(x) == i]]

    # make plots
    for condition, legend_plot, date_plot, name_plot in zip(ld_rep_paths, legend_plot, dates_plot, name_plot):
        # create path_out containing dates of first and last folder in which replicates were found
        path_out = os.path.join(params['path_out'], date_plot[0])
        if len(date_plot) > 1:
            path_out = f'{path_out}-{date_plot[-1]}'
        path_out = f"{path_out}_NrPar{type_plot}_{location}_{ch_foci}"

        # plot data
        if params['plotCom'] == 0:
            # select only parts of data corresponding to the condition ld_rep_paths.keys()[i] to make plots
            df_plot = df_all.loc[condition].reset_index()
        elif params['plotCom'] == 1:
            # select all data to make plots
            df_plot = df_all.reset_index()

        df_plot.to_csv(f"{path_out}_df_plot_{name_plot}.txt", sep='\t')  # save plot df

        # --- Parameter plots ---
        for par in plt_params:
            print(f'plotNrPar {par}')
            # get y_max
            par_max = np.max(df_plot[par])
            y_max = get_y_max(par_max)
            if params['plotPar']['y_max']:
                y_max = params['plotPar']['y_max']

            # Make jitterplot
            fig, axs = plt.subplots()
            stripplot = sns.stripplot(data=df_plot, x='Nr', y=par, hue=sort_plot, palette=color_plot, alpha=0.5, jitter=True, dodge=True)
            sns.boxplot(data=df_plot, x='Nr', y=par, hue=sort_plot, color='lightgrey', showfliers=False)

            # Get locations of legend labels to plot only stripplot legend
            if params['legend'] == 1:
                handles, labels = stripplot.get_legend_handles_labels()
                leg = plt.legend(handles[:len(legend_plot):], legend_plot, bbox_to_anchor=(1, 0.5), loc='center left',
                                 markerscale=1)
                for lh in leg.legendHandles:
                    lh.set_alpha(1)

            plt.ylabel(f"{label}\n")
            plt.ylim(params['plotPar']['y_min'], y_max)
            # plt.locator_params(axis="y", nbins=8)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(-2, 0))
            plt.xlabel(f"Count per {location}")
            plt.savefig(f"{path_out}_{par}_{name_plot}{params['file_type']}", bbox_inches='tight')
            plt.close(fig)

            # create metadata file with information about pipeline version and replicates used
            with open(os.path.abspath(_version.__file__)) as f:
                txt_out = f.read()
            txt_out += '\nReplicates analyzed:\n' + '\n'.join(np.hstack(legend_sep))
            np.savetxt(f"{path_out}_{par}_{name_plot}.txt", [txt_out], fmt='%s')


def df_xyz_to_um(df, xy_um=0.09707, z_um=0.25):
    if xy_um:
        df['x_um'] = df['x'] * xy_um
        df['y_um'] = df['y'] * xy_um
    if z_um:
        df['z_um'] = df['z'] * z_um
    return df


def plot_dis(params, ld_rep_paths, name_plot):
    """
    Make distance plots
    """
    print('plotDis')

    # plot text
    legend_com = []  # list of legendNames for each condition including number of replicates
    legend_sep = []  # list with for each condition a list with legend names from each replicate
    dates_all = []  # list of dates from each replicate within a condition

    # plot data
    ch_ref = params['plotDis']['ch_ref']
    ch_foci = 'ch2' if ch_ref == 'ch1' else 'ch1'
    pixel_size_xy = params['pixel_size_xy']
    pixel_size_z = params['pixel_size_z']
    plt_params = params['plt_params']
    cols = np.unique(['Ip', 'R2_peak', 'z', 'y', 'x', 's', 'dx', 'dy', 'dz', 'fit', 'cell', 'nucleus', 'file'] + plt_params)
    df_all = pd.DataFrame()

    for condition, replicates in ld_rep_paths.items():
        rep_names = []  # List with replicate names
        rep_dates = []  # List with dates (folders) from which data was used
        cell_nr_com = 0  # Number of cells in condition

        for rep_ID in replicates:
            path_in, rep_name, cell_nr = rep_ID

            # plot text
            rep_names.append(f"{path_in.split('/')[-1]} {rep_name.replace('_', ' ')} ({cell_nr} cells)")
            rep_dates.append(path_in.split('/')[-1])
            cell_nr_com += cell_nr

            # load cellular foci which are fit
            loc_foci = pd.read_csv(f"{path_in}_loc_results_{ch_foci}_{rep_name}.txt", sep='\t', usecols=cols).query(
                f"fit==True & cell>0 & R2_peak>-1").drop(columns=['fit', 'nucleus']).set_index(['file', 'cell'])
            # load nuclear ref foci which are fit and filter on the maximum Ip per nucleus to get the ref foci
            loc_ref = pd.read_csv(f"{path_in}_loc_results_{ch_ref}_{rep_name}.txt", sep='\t', usecols=cols).query(
                                  f"fit==True & nucleus>0 & R2_peak>-1").drop(columns=['fit', 'nucleus'])
            loc_ref = loc_ref.groupby(['file', 'cell']).apply(lambda x: loc_ref.loc[x['Ip'].idxmax()]).set_index(['file', 'cell'])
            # load random location cells (= contain all cells)
            loc_ctrl = pd.read_csv(f"{path_in}_loc_random_{rep_name}.txt", sep='\t').set_index(['file', 'cell'])

            # calculate coordinates in um
            loc_foci = df_xyz_to_um(loc_foci, pixel_size_xy, pixel_size_z).drop(columns=['x', 'y', 'z'])
            loc_ref = df_xyz_to_um(loc_ref, pixel_size_xy, pixel_size_z).drop(columns=['x', 'y', 'z'])
            loc_ctrl = df_xyz_to_um(loc_ctrl, pixel_size_xy, None).drop(columns=['x', 'y'])
            loc_ctrl.rename(columns={"x_um": "x_um_RDM", "y_um": "y_um_RDM"}, inplace=True)

            # combine all replicate dataframes
            loc_rep = loc_foci.join(loc_ref, lsuffix=f"_{ch_foci}", rsuffix=f"_{ch_ref}", how='outer')
            loc_rep = loc_rep.join(loc_ctrl, how='outer').reset_index()

            loc_rep['replicate'] = f"{path_in.split('/')[-1]}_{rep_name}"
            loc_rep['condition'] = condition

            # add loc_rep dataframe to loc_com dataframe
            df_all = pd.concat((df_all, loc_rep), ignore_index=True)

        rep_nr = len(replicates)  # number of replicates in condition
        legend_com.append(f'{condition} (n = {rep_nr}, {cell_nr_com} cells, ')
        legend_sep.append(rep_names)
        dates_all.append([x for i, x in enumerate(rep_dates) if rep_dates.index(x) == i])

    # set df_plot indexes to condition in order to select replicates from same condition to make plots later on
    df_all = df_all.set_index(['condition', 'replicate'])

    # calculate sigma xy in um
    if 's' in plt_params:
        print('calculating sigma xy in um')
        df_all[f's_um_{ch_ref}'] = df_all[f's_{ch_ref}'] * params['pixel_size_xy']
        df_all[f's_um_{ch_foci}'] = df_all[f's_{ch_foci}'] * params['pixel_size_xy']
        plt_params = list(map(lambda x: x.replace('s', 's_um'), plt_params))
    if 'sz' in plt_params:
        print('calculating sigma z in um')
        df_all[f'sz_um_{ch_ref}'] = df_all[f'sz_{ch_ref}'] * params['pixel_size_z']
        df_all[f'sz_um_{ch_foci}'] = df_all[f'sz_{ch_foci}'] * params['pixel_size_z']
        plt_params = list(map(lambda x: x.replace('sz', 'sz_um'), plt_params))

    # set plot labels/legends
    foci_name = params['foci_name']
    ref_name = params['plotDis']['ref_name']
    threshold = params['plotDis']['threshold']
    color_plot = params['color_palette']

    # calculate distances
    df_all[f"{foci_name}-{ref_name} 3D"] = ((df_all[f"x_um_{ch_ref}"] - df_all[f"x_um_{ch_foci}"]) ** 2 +
                                            (df_all[f"y_um_{ch_ref}"] - df_all[f"y_um_{ch_foci}"]) ** 2 +
                                            (df_all[f"z_um_{ch_ref}"] - df_all[f"z_um_{ch_foci}"]) ** 2) ** 0.5
    df_all[f"{foci_name}-{ref_name} 2D"] = ((df_all[f"x_um_{ch_ref}"] - df_all[f"x_um_{ch_foci}"]) ** 2 +
                                            (df_all[f"y_um_{ch_ref}"] - df_all[f"y_um_{ch_foci}"]) ** 2) ** 0.5
    df_all[f"{foci_name}-RDM 2D"] = ((df_all[f"x_um_RDM"] - df_all[f"x_um_{ch_foci}"]) ** 2 +
                                     (df_all[f"y_um_RDM"] - df_all[f"y_um_{ch_foci}"]) ** 2) ** 0.5

    # # calculate foci-edge to ref-center 3D distance
    # df_all[f"{foci_name}-{ref_name} edge-center 3D"] = df_all[f"{foci_name}-{ref_name} 3D"] - df_all[f's_{ch_foci}']*params['plotDis']['pixel_size_xy']

    # set distance bins
    dis_bins = np.r_[0:3/params['plotDis']['binSize'] * params['plotDis']['binSize']:params['plotDis']['binSize']]
    dis_bin_centre = 0.5 * (dis_bins[1:] + dis_bins[:-1])  # Use centre x-position instead of bin edge for plotting

    # calculate step size of yticks for frequency plots using a fixed tick number and 1 decimal
    ystep = 0
    ytick_nr = 4
    while ystep == 0:
        ytick_nr += -1
        ystep = np.round(params['plotDis']['y_max_frq']/ytick_nr, 1)

    if params['plotCom'] == 0:
        type_plot = 'Sep'
        sort_plot = 'replicate'
        legend_plot = legend_sep
        dates_plot = dates_all
    elif params['plotCom'] == 1:
        type_plot = 'Com'
        sort_plot = 'condition'
        legend_plot = [legend_com]
        dates_plot = [[x for i, x in enumerate(dates_all[0]) if dates_all[0].index(x) == i]]

    # make plots
    for condition, legend_plot, date_plot, name_plot in zip(ld_rep_paths, legend_plot, dates_plot, name_plot):
        # create path_out containing dates of first and last folder in which replicates were found
        path_out = os.path.join(params['path_out'], date_plot[0])
        if len(date_plot) > 1:
            path_out = f'{path_out}-{date_plot[-1]}'
        path_out = f'{path_out}_Dis{type_plot}'

        # create metadata file with information about pipeline version and replicates used
        with open(os.path.abspath(_version.__file__)) as f:
            txt_out = f.read()
        txt_out += '\nReplicates analyzed:\n' + '\n'.join(np.hstack(legend_sep))
        np.savetxt(f"{path_out}_{name_plot}.txt", [txt_out], fmt='%s')

        # plot data
        if params['plotCom'] == 0:
            # select only parts of data corresponding to the condition ld_rep_paths.keys()[i] to make plots
            df_plot = df_all.loc[condition]
        elif params['plotCom'] == 1:
            # select all data to make plots
            df_plot = df_all

        df_plot.to_csv(f"{path_out}_{name_plot}_df_plot.txt", sep='\t')  # save plot df
        sort_values = df_plot.index.unique(level=sort_plot).tolist()  # unique replicates/conditions in order of conditions_to_analyze
        df_plot.sort_index(inplace=True)  # sort df_plot to speed up df_plot.loc[]

        # set plot height
        con_nrs = len(sort_values)
        height = scalar/2
        if con_nrs == 1:
            height = scalar

        col_int_ref = f"Ip_{ch_ref}"  # ref intensity column name
        col_int_foci = f"Ip_{ch_foci}"  # foci intensity column name

        # --- Foci-TS/DNA label distance plots ---
        distances = {}  # create dictionary with all distances to plot
        if params['plotDis']['plot_3D'] == 1:
            distances['3D'] = {'col_dis': f"{foci_name}-{ref_name} 3D",  # 3D distances between foci and ref
                               'x_label': f"{foci_name} - {ref_name}\n3D"}
        if params['plotDis']['plot_2D'] == 1:
            distances['2D'] = {'col_dis': f"{foci_name}-{ref_name} 2D",  # 2D distances between foci and ref
                               'x_label': f"{foci_name} - {ref_name}\n2D"}
        if params['plotDis']['plot_RDM'] == 1:
            distances['RDM'] = {'col_dis': f"{foci_name}-RDM 2D",  # 2D distances between foci and random nuclear spot
                                'x_label': f"{foci_name} - RDM\n2D"}

        for dis in distances.keys():
            col_dis = distances[dis]['col_dis']
            x_label = distances[dis]['x_label']
            print(f"plotDis: {col_dis}")

            # --- Hist all distances ---
            fig, axs = plt.subplots(figsize=(scalar, height * con_nrs), ncols=1, nrows=con_nrs, sharex=True)

            for i, ax in enumerate(fig.axes):
                con_name = sort_values[i]
                rep_names = df_plot.loc[[con_name]].index.unique(level='replicate').tolist()

                # loop over replicates and get cell nrs and distance data
                con_data = []  # nested list with rep distances
                rep_weights = []  # cell number per replicate used for normalization
                for rep_name in rep_names:
                    data_plt = df_plot.loc[(con_name, rep_name), ['file', 'cell', col_dis, col_int_ref]].query(f"{col_int_ref}>0")  # filter rows which have a ref foci
                    rep_cell_nr = len(data_plt.groupby(['file', 'cell']))  # all cells which have a ref foci
                    rep_weights.append(rep_cell_nr)
                    rep_data = data_plt[col_dis].to_numpy() # all distances
                    con_data.append(rep_data)

                # plot weighted mean and weighted standard deviation as error in case of multiple replicates
                n_data = [len(rep) for rep in con_data]  # number of data points per rep
                n_cells = rep_weights  # number of cells with a reference (TS or DNA label) per rep
                con_weights = weights(n_data, n_cells)
                con_hist, con_sem = bootstrap(np.concatenate(con_data), con_weights, dis_bins, norm=sum(n_cells))

                rep_nrs = len(con_data)  # number of replicates of current condition
                if rep_nrs > 1:
                    ax.fill_between(dis_bin_centre, con_hist + con_sem, con_hist - con_sem, color=color_plot[i],
                                    alpha=alpha, linewidth=0)
                text_n_data = f"{len(np.concatenate(con_data))} distances)"
                ax.plot(dis_bin_centre, con_hist, color=color_plot[i], label=legend_plot[i] + text_n_data)

                # plot axis
                ax.set_xlim(0, params['plotDis']['x_max'])
                ax.set_xticks(np.arange(0, params['plotDis']['x_max']+.1, step=0.4))
                if i == con_nrs-1:
                    ax.set_xlabel(f"{x_label} distance (μm)")
                ax.set_ylim(0, params['plotDis']['y_max_frq'])
                ax.set_yticks(np.arange(0, params['plotDis']['y_max_frq']+0.01, step=ystep))
                if params['legend'] == 1:
                    ax.legend(bbox_to_anchor=(1, 0.5), loc='center left',
                              fontsize='large', markerscale=2)

            # plot threshold vertical -- line and add legends
            for i, ax in enumerate(fig.axes):
                # plot threshold vertical -- line
                ax.vlines(threshold, 0, params['plotDis']['y_max_frq'], colors=ecolor, linestyles='--')
            fig.supylabel('fraction', x=-0.08)
            plt.savefig(f"{path_out}_{name_plot}_{dis}_hist_all{params['file_type']}", bbox_inches='tight')
            plt.close(fig)

            # --- Hist Nearest Neighbor Distance ---
            fig, axs = plt.subplots(figsize=(scalar, height * con_nrs), ncols=1, nrows=con_nrs, sharex=True)
            for i, ax in enumerate(fig.axes):
                con_name = sort_values[i]
                rep_names = df_plot.loc[[con_name]].index.unique(level='replicate').tolist()

                # loop over replicates and get cell nrs and NND data
                con_data = []  # nested list with rep NNDs
                rep_weights = []  # cell number per replicate used for normalization
                for rep_name in rep_names:
                    data_plt = df_plot.loc[(con_name, rep_name), ['file', 'cell', col_dis, col_int_ref]].query(f"{col_int_ref}>0")  # filter rows which have a ref foci
                    rep_cell_nr = len(data_plt.groupby(['file', 'cell']))  # all cells which have a ref foci
                    rep_weights.append(rep_cell_nr)
                    rep_data = data_plt.groupby(['file', 'cell'])[col_dis].min().to_numpy()  # minimal distances for each cell
                    con_data.append(rep_data)

                # plot weighted mean and weighted standard deviation as error in case of multiple replicates
                n_data = [len(rep) for rep in con_data]  # number of data points per rep
                n_cells = rep_weights  # number of cells with a reference (TS or DNA label) per rep
                con_weights = weights(n_data, n_cells)
                con_hist, con_sem = bootstrap(np.concatenate(con_data), con_weights, dis_bins, norm=sum(n_cells))

                rep_nrs = len(con_data)  # number of replicates of current condition
                if rep_nrs > 1:
                    ax.fill_between(dis_bin_centre, con_hist + con_sem, con_hist - con_sem, color=color_plot[i],
                                    alpha=alpha, linewidth=0)

                text_n_data = f"{len(np.concatenate(con_data))} distances)"
                ax.plot(dis_bin_centre, con_hist, color=color_plot[i], label=legend_plot[i] + text_n_data)

                # Calculate percentage of ref-foci 3D distances below threshold
                thr_per, thr_err = bootstrap(np.concatenate(con_data), con_weights,
                                             [0, threshold, max(np.concatenate(con_data))], norm=sum(n_cells))
                ax.text(0, params['plotDis']['y_max_frq']*1.025,
                        f'{thr_per[0]*100:.1f} ± {thr_err[0]*100:.1f}%', ha='left')

                # plot axis
                ax.set_xlim(0, params['plotDis']['x_max'])
                ax.set_xticks(np.arange(0, params['plotDis']['x_max']+.1, step=0.4))
                if i == con_nrs-1:
                    ax.set_xlabel(f"{x_label} NND (μm)")
                ax.set_ylim(0, params['plotDis']['y_max_frq'])
                ax.set_yticks(np.arange(0, params['plotDis']['y_max_frq']+0.01, step=ystep))
                if params['legend'] == 1:
                    ax.legend(bbox_to_anchor=(1, 0.5), loc='center left',
                              fontsize='large', markerscale=2)

            # plot threshold vertical -- line and add legends
            for i, ax in enumerate(fig.axes):
                # plot threshold vertical -- line
                ax.vlines(threshold, 0, params['plotDis']['y_max_frq'], colors=ecolor, linestyles='--')

            fig.supylabel('fraction', x=-0.08)
            plt.savefig(f"{path_out}_{name_plot}_{dis}_hist_nearest{params['file_type']}", bbox_inches='tight')
            plt.close(fig)

            # --- Hist Nearest Neighbor Distance in one plot ---
            fig, axs = plt.subplots(figsize=(scalar, scalar*0.8), ncols=1, nrows=1, sharex=True)

            for i in range(len(sort_values)):
                con_name = sort_values[i]
                rep_names = df_plot.loc[[con_name]].index.unique(level='replicate').tolist()

                # loop over replicates and get cell nrs and NND data
                con_data = []  # nested list with rep NNDs
                rep_weights = []  # cell number per replicate used for normalization
                for rep_name in rep_names:
                    data_plt = df_plot.loc[(con_name, rep_name), ['file', 'cell', col_dis, col_int_ref]].query(f"{col_int_ref}>0")  # filter rows which have a ref foci
                    rep_cell_nr = len(data_plt.groupby(['file', 'cell']))  # all cells which have a ref foci
                    rep_weights.append(rep_cell_nr)
                    rep_data = data_plt.groupby(['file', 'cell'])[col_dis].min().to_numpy()  # minimal distances for each cell
                    con_data.append(rep_data)

                # plot weighted mean and weighted standard deviation as error in case of multiple replicates
                n_data = [len(rep) for rep in con_data]  # number of data points per rep
                n_cells = rep_weights  # number of cells with a reference (TS or DNA label) per rep
                con_weights = weights(n_data, n_cells)
                con_hist, con_sem = bootstrap(np.concatenate(con_data), con_weights, dis_bins, norm=sum(n_cells))

                rep_nrs = len(con_data)  # number of replicates of current condition
                if rep_nrs > 1:
                    axs.fill_between(dis_bin_centre, con_hist + con_sem, con_hist - con_sem, color=color_plot[i],
                                     alpha=alpha, linewidth=0)

                text_n_data = f"{len(np.concatenate(con_data))} distances)"
                axs.plot(dis_bin_centre, con_hist, color=color_plot[i], label=legend_plot[i] + text_n_data)

                # Calculate percentage of ref-foci 3D distances below threshold
                # thr_per, thr_err = bootstrap(np.concatenate(con_data), con_weights,
                #                              [0, threshold, max(np.concatenate(con_data))], norm=sum(n_cells))
                # ax.text(0, params['plotDis']['y_max_frq']*1.025,
                #         f'{thr_per[0]*100:.1f} ± {thr_err[0]*100:.1f}%', ha='left')

            # plot axis
            axs.set_xlim(0, params['plotDis']['x_max'])
            axs.set_xticks(np.arange(0, params['plotDis']['x_max']+.1, step=0.4))
            axs.set_xlabel(f"{x_label} NND (μm)")
            axs.set_ylim(0, params['plotDis']['y_max_frq'])
            axs.set_yticks(np.arange(0, params['plotDis']['y_max_frq']+0.01, step=ystep))

            # plot threshold vertical -- line and add legends
            axs.vlines(threshold, 0, params['plotDis']['y_max_frq'], colors=ecolor, linestyles='--')
            if params['legend'] == 1:
                axs.legend(bbox_to_anchor=(1, 0.5), loc='center left',
                           fontsize='large', markerscale=2)
            fig.supylabel('fraction', x=-0.08)
            plt.savefig(f"{path_out}_{name_plot}_{dis}_hist_nearest2{params['file_type']}", bbox_inches='tight')
            plt.close(fig)

            # --- Barplot 3D overlap fraction ---
            fig, axs = plt.subplots(figsize=(0.25*con_nrs, 1))
            fisher_exact = []  # list with for each condition counts of (#total cells with ref, #overlapping cells)
            text_n_data = []
            for i in range(len(sort_values)):
                con_name = sort_values[i]
                rep_names = df_plot.loc[[con_name]].index.unique(level='replicate').tolist()

                # loop over replicates and get cell nrs and NND data
                con_data = []  # nested list with rep NNDs
                rep_weights = []  # cell number per replicate used for normalization
                for rep_name in rep_names:
                    data_plt = df_plot.loc[(con_name, rep_name), ['file', 'cell', col_dis, col_int_ref]].query(f"{col_int_ref}>0")  # filter rows which have a ref foci
                    rep_cell_nr = len(data_plt.groupby(['file', 'cell']))  # all cells which have a ref foci
                    rep_weights.append(rep_cell_nr)
                    rep_data = data_plt.groupby(['file', 'cell'])[col_dis].min().to_numpy()  # minimal distances for each cell, no ref = nan
                    con_data.append(rep_data)

                # get weighted mean and weighted standard deviation as error in case of multiple replicates
                n_data = [len(rep) for rep in con_data]  # number of data points per rep
                n_cells = rep_weights  # number of cells with a reference (TS or DNA label) per rep
                con_weights = weights(n_data, n_cells)  # should be equal to 1 as every cell included has a ref and foci

                # calculate percentage of ref-foci 3D distances below threshold
                con_data_combined = np.concatenate(con_data)
                thr_per, thr_err = bootstrap(con_data_combined, con_weights,
                                             [0, threshold, max(con_data_combined)], norm=sum(n_cells))
                text_n_data.append(f"{len(np.concatenate(con_data))} label cells)")
                axs.bar(i, thr_per[0], yerr=thr_err[0], error_kw=dict(ecolor=ecolor, capsize=2), color=color_plot[i], label=legend_plot[i])

                # Fishers exact test to test whether amount of overlap differs between conditions
                n_cells_condition = len(con_data_combined)
                n_cells_overlap = len([x for x in con_data_combined if x < threshold])
                fisher_exact += [[n_cells_condition-n_cells_overlap, n_cells_overlap]]

            axs.get_xaxis().set_ticks([])
            axs.set_ylim(0, params['plotDis']['y_max_bar'])
            axs.set_ylabel(f"overlap fraction ")

            if params['legend'] == 1:
                plt.legend([a + b for a, b in zip(legend_plot, text_n_data)], bbox_to_anchor=(1, 0.5), loc='center left', markerscale=2)
            plt.savefig(f"{path_out}_{name_plot}_{dis}_bar_overlap{params['file_type']}", bbox_inches='tight')
            plt.close(fig)

            if con_nrs > 1:
                # compute the Fisher Exact Test and save as txt file
                text = 'Test differences in amount of cells with a reference which have an overlapping foci\n\n'
                for item in legend_com:
                    text += f"{legend_com.index(item)+1}) {item}\n"
                text += '\nFisher exact test (two-sided, without p-value correction)\n'

                fisher_exact_results = []
                for i in range(len(fisher_exact)):
                    fisher_exact_results.append([ss.fisher_exact([fisher_exact[i], fisher_exact[j]])[1] for j in range(len(fisher_exact))])
                indexes = np.arange(1, con_nrs+1, 1)
                text += pd.DataFrame(fisher_exact_results, columns=indexes, index=indexes).to_string()
                print(text)
                np.savetxt(f"{path_out}_{name_plot}_{dis}_bar_overlap.txt", [text], fmt='%s')

            # --- Piechart fractions 3D all ---
            fig, axs = plt.subplots(figsize=(scalar, scalar * con_nrs), ncols=1, nrows=con_nrs, sharex=True)
            labels = f'{ref_name} only', 'empty', f'{foci_name} only', f'{ref_name} + {foci_name}\nno overlap', f'{ref_name} + {foci_name}\noverlap',
            colors = ['#AA4499', '#A9A9A9', '#117733', '#91CCC2', 'white']

            for i, ax in enumerate(fig.axes):
                con_name = sort_values[i]
                rep_names = df_plot.loc[[con_name]].index.unique(level='replicate').tolist()

                # loop over replicates and get cell nrs and NND data
                con_counts = np.array([0, 0, 0, 0, 0])  # array with counts of con_non_overlap, con_overlapping, con_foci_only, con_ref_only, con_empty
                for rep_name in rep_names:
                    data_plot = df_plot.loc[(con_name, rep_name), ['file', 'cell', col_dis, col_int_ref, col_int_foci]]
                    data_plot_grouped = data_plot.groupby(['file', 'cell'])[[col_int_foci, col_int_ref]]
                    rep_empty = len(data_plot_grouped.apply('max').query(f"{col_int_foci}.isnull() & {col_int_ref}.isnull()"))
                    rep_ref_only = len(data_plot_grouped.apply('max').query(f"{col_int_foci}.isnull() & {col_int_ref}>0"))
                    rep_foci_only = len(data_plot_grouped.apply('max').query(f"{col_int_foci}>0 & {col_int_ref}.isnull()"))

                    rep_dis = data_plot.groupby(['file', 'cell'])[col_dis].apply(lambda x: x.min()).to_list()
                    rep_overlapping = len([x for x in rep_dis if x<threshold])
                    rep_non_overlap = len([x for x in rep_dis if x>=threshold])

                    con_counts += np.array([rep_ref_only, rep_empty, rep_foci_only, rep_non_overlap, rep_overlapping])

                # Bootstrapping standard error of the mean
                data = np.concatenate([[i]*con_counts[i] for i in range(len(con_counts))])
                w = len(data)*[1]
                con_hist, con_sem = bootstrap(data, w, bins=[0, 1, 2, 3, 4, 5])

                _wedges, lables, percentages = \
                    ax.pie(con_counts, labels=labels, autopct='%1.1f%%', pctdistance=0.7, labeldistance=1.2,
                           startangle=90, colors=colors, textprops={'fontsize': 9},
                           wedgeprops={"edgecolor" : "black", 'linewidth': 1, 'antialiased': True}
                       )
                for pct, frc, sem in zip(percentages, con_hist, con_sem):
                    pct.set_text(f"{int(np.round(frc*100, 0))}±{int(np.round(sem*100, 0))}%")
                if params['legend'] == 1:
                    ax.text(3, 0, legend_plot[i][:-2]+')')
            plt.savefig(f"{path_out}_{name_plot}_{dis}_pie_all{params['file_type']}", bbox_inches='tight')
            plt.close(fig)

            # --- Piechart fractions 3D ref cells only ---
            fig, axs = plt.subplots(figsize=(scalar, scalar * con_nrs), ncols=1, nrows=con_nrs, sharex=True)
            labels = f'no {foci_name}', f'{foci_name}, no overlap', f'{foci_name}, overlap',
            colors = ['#AA4499', '#117733', 'white']
            for i, ax in enumerate(fig.axes):
                con_name = sort_values[i]
                rep_names = df_plot.loc[[con_name]].index.unique(level='replicate').tolist()

                # loop over replicates and get cell nrs and NND data
                con_counts = np.array([0, 0, 0])  # array with counts of con_non_overlap, con_overlapping, con_foci_only, con_ref_only, con_empty
                for rep_name in rep_names:
                    data_plot = df_plot.loc[(con_name, rep_name), ['file', 'cell', col_dis, col_int_ref, col_int_foci]]
                    data_plot_grouped = data_plot.groupby(['file', 'cell'])[[col_int_foci, col_int_ref]]
                    # rep_empty = len(data_plot_grouped.apply('max').query(f"{col_int_foci}.isnull() & {col_int_ref}.isnull()"))
                    rep_ref_only = len(data_plot_grouped.apply('max').query(f"{col_int_foci}.isnull() & {col_int_ref}>0"))
                    # rep_foci_only = len(data_plot_grouped.apply('max').query(f"{col_int_foci}>0 & {col_int_ref}.isnull()"))

                    rep_dis = data_plot.groupby(['file', 'cell'])[col_dis].apply(lambda x: x.min()).to_list()
                    rep_overlapping = len([x for x in rep_dis if x<threshold])
                    rep_non_overlap = len([x for x in rep_dis if x>=threshold])

                    con_counts += np.array([rep_ref_only, rep_non_overlap, rep_overlapping])
                # Bootstrapping standard error of the mean
                data = np.concatenate([[i]*con_counts[i] for i in range(len(con_counts))])
                w = len(data)*[1]
                con_hist, con_sem = bootstrap(data, w, bins=[0, 1, 2, 3])

                _wedges, lables, percentages = \
                    ax.pie(con_counts, labels=labels, autopct='%1.1f%%', pctdistance=0.7, labeldistance=1.2,
                           startangle=90, colors=colors, textprops={'fontsize': 9},
                           wedgeprops={"edgecolor" : "black", 'linewidth': 1, 'antialiased': True}
                           )
                for pct, frc, sem in zip(percentages, con_hist, con_sem):
                    pct.set_text(f"{int(np.round(frc*100, 0))}±{int(np.round(sem*100, 0))}%")
                if params['legend'] == 1:
                    ax.text(3, 0, legend_plot[i][:-2]+')')
            plt.savefig(f"{path_out}_{name_plot}_{dis}_pie_ref{params['file_type']}", bbox_inches='tight')
            plt.close(fig)

            # # --- Barplot fractions 3D ref ---
            # fig, axs = plt.subplots(figsize=(scalar/6, height * con_nrs), ncols=1, nrows=con_nrs, sharex=True)
            #
            # for i, ax in enumerate(fig.axes):
            #     con_name = sort_values[i]
            #     rep_names = df_plot.loc[[con_name]].index.unique(level='replicate').tolist()
            #
            #     # loop over replicates and get cell nrs and NND data
            #     con_counts = np.array([0, 0, 0])  # array with counts of con_overlapping, con_non_overlap, con_labels_only
            #     for rep_name in rep_names:
            #         data_plot = df_plot.loc[(con_name, rep_name), ['file', 'cell', col_dis, col_int_ref, col_int_foci]].query(f"{col_int_ref}>0")  # filter rows which have a ref foci
            #         rep_ref_only = len(data_plot.query(f"{col_int_foci}!={col_int_foci}").groupby(['file', 'cell']))
            #         rep_dis = data_plot.groupby(['file', 'cell'])[f"{col_dis}"].apply(lambda x: x.min()).to_list()
            #         rep_overlapping = len([x for x in rep_dis if x<threshold])
            #         rep_non_overlap = len([x for x in rep_dis if x>=threshold])
            #
            #         con_counts += np.array([rep_overlapping, rep_non_overlap, rep_ref_only])
            #
            #     con_freq = con_counts / sum(con_counts)
            #
            #     # Bootstrapping weighted SD
            #     data = np.concatenate([[i]*con_counts[i] for i in range(len(con_counts))])
            #     w = len(data)*[1]
            #     con_hist, con_sem = bootstrap(data, w, bins=[0, 1, 2, 3])
            #
            #     ax.bar(1, con_hist[2], color='#f768a1', yerr=con_sem[2], error_kw=dict(ecolor=ecolor, capsize=5), bottom=con_freq[0]+con_freq[1], label='Label only')
            #     ax.bar(1, con_hist[1], color='#99d8c9', yerr=con_sem[1], error_kw=dict(ecolor=ecolor, capsize=5), bottom=con_freq[0], label='No overlap')
            #     ax.bar(1, con_hist[0], color='#f7fcfd', yerr=con_sem[0], error_kw=dict(ecolor=ecolor, capsize=5), bottom=0, label='Overlap')
            #     ax.tick_params(
            #         axis='x',           # changes apply to the x-axis
            #         which='both',       # both major and minor ticks are affected
            #         bottom=False,       # ticks along the bottom edge are off
            #         top=False,          # ticks along the top edge are off
            #         labelbottom=False)  # labels along the bottom edge are off
            #     ax.legend(bbox_to_anchor=(1, 0.5), loc='center left', markerscale=2)
            #     if params['legend'] == 1:
            #         ax.text(5, 0.5, legend_plot[i])
            #
            # fig.supylabel('fraction', x=-0.9)
            # plt.savefig(f"{path_out}_{name_plot}_{dis}_fraction_ref{params['file_type']}", bbox_inches='tight')
            # plt.close(fig)
            #
            # # --- Barplot fractions 3D all ---
            # fig, axs = plt.subplots(figsize=(scalar/6, height * con_nrs), ncols=1, nrows=con_nrs, sharex=True)
            # for i, ax in enumerate(fig.axes):
            #     con_name = sort_values[i]
            #     rep_names = df_plot.loc[[con_name]].index.unique(level='replicate').tolist()
            #
            #     # loop over replicates and get cell nrs and NND data
            #     con_counts = np.array([0, 0, 0, 0, 0])  # array with counts of con_non_overlap, con_overlapping, con_foci_only, con_ref_only, con_empty
            #     for rep_name in rep_names:
            #         data_plot = df_plot.loc[(con_name, rep_name), ['file', 'cell', col_dis, col_int_ref, col_int_foci]]
            #         data_plot_grouped = data_plot.groupby(['file', 'cell'])[[col_int_foci, col_int_ref]]
            #         rep_empty = len(data_plot_grouped.apply('max').query(f"{col_int_foci}.isnull() & {col_int_ref}.isnull()"))
            #         rep_ref_only = len(data_plot_grouped.apply('max').query(f"{col_int_foci}.isnull() & {col_int_ref}>0"))
            #         rep_foci_only = len(data_plot_grouped.apply('max').query(f"{col_int_foci}>0 & {col_int_ref}.isnull()"))
            #
            #         rep_dis = data_plot.groupby(['file', 'cell'])[col_dis].apply(lambda x: x.min()).to_list()
            #         rep_overlapping = len([x for x in rep_dis if x<threshold])
            #         rep_non_overlap = len([x for x in rep_dis if x>=threshold])
            #
            #         con_counts += np.array([rep_non_overlap, rep_overlapping, rep_foci_only, rep_ref_only, rep_empty])
            #
            #     con_freq = con_counts / sum(con_counts)
            #
            #     # Bootstrapping weighted SD
            #     data = np.concatenate([[i]*con_counts[i] for i in range(len(con_counts))])
            #     w = len(data)*[1]
            #     con_hist, con_sem = bootstrap(data, w, bins=[0, 1, 2, 3, 4, 5, 6])
            #     ax.bar(1, con_hist[4], color='lightgrey', yerr=con_sem[4], error_kw=dict(ecolor=ecolor, capsize=5), bottom=sum(con_freq[:4]), label='empty')
            #     ax.bar(1, con_hist[3], color='lightblue', yerr=con_sem[3], error_kw=dict(ecolor=ecolor, capsize=5), bottom=sum(con_freq[:3]), label=f'{ref_name} only')
            #     ax.bar(1, con_hist[2], color='#99d8c9', yerr=con_sem[2], error_kw=dict(ecolor=ecolor, capsize=5), bottom=sum(con_freq[:2]), label=f'{foci_name} only')
            #     ax.bar(1, con_hist[1], color='#f768a1', yerr=con_sem[1], error_kw=dict(ecolor=ecolor, capsize=5), bottom=sum(con_freq[:1]), label='No overlap')
            #     ax.bar(1, con_hist[0], color='#f7fcfd', yerr=con_sem[0], error_kw=dict(ecolor=ecolor, capsize=5), bottom=0, label='Overlap')
            #
            #     ax.tick_params(
            #         axis='x',           # changes apply to the x-axis
            #         which='both',       # both major and minor ticks are affected
            #         bottom=False,       # ticks along the bottom edge are off
            #         top=False,          # ticks along the top edge are off
            #         labelbottom=False)  # labels along the bottom edge are off
            #     ax.legend(bbox_to_anchor=(1, 0.5), loc='center left',
            #               fontsize='medium', markerscale=1)
            #     if params['legend'] == 1:
            #         ax.text(5, 0.5, legend_plot[i])
            #
            # fig.supylabel('fraction', x=-0.9)
            # plt.savefig(f"{path_out}_{name_plot}_{dis}_fraction_all{params['file_type']}", bbox_inches='tight')
            # plt.close(fig)

        if params['plotDis']['plot_params'] == 1:
            # --- Parameter plots ---
            for p in range(len(plt_params)):
                par = plt_params[p]
                par_ref = f"{par}_{ch_ref}"  # ref parameter column name
                par_foci = f"{par}_{ch_foci}"  # foci parameter column name
                print(f'plotPar: {par}')

                label = params['plt_params_labels'][p]
                y_min = params['plt_params_y_min'][p]
                y_max = params['plt_params_y_max'][p]
                y_max_ref = y_max_foci = y_max

                if not y_max:
                    par_max_ref = np.max(df_plot[par_ref])
                    y_max_ref = get_y_max(par_max_ref)

                    par_max_foci = np.max(df_plot[par_foci])
                    y_max_foci = get_y_max(par_max_foci)

                # get whether to use 2D or 3D distances between foci and ref
                xD = params['plotDis']['plot_par_xD']
                col_dis = f"{foci_name}-{ref_name} {xD}"

                # --- Scat 3D distance vs foci param ---
                if params['plotDis']['plot_par_xD'] == '3D':
                    fig, axs = plt.subplots(figsize=(scalar, height * con_nrs), ncols=1, nrows=con_nrs, sharex=True)

                    for i, ax in enumerate(fig.axes):
                        con_name = sort_values[i]

                        # plot data
                        data_plt = df_plot.loc[con_name, [col_dis, par_foci]].dropna()
                        data_dis = data_plt[col_dis].to_numpy()
                        data_int = data_plt[par_foci].to_numpy()
                        text_n_data = f"{len(data_int)} spots)"
                        ax.scatter(data_dis, data_int, c=color_plot[i], alpha=alpha, s=(scalar+1)**2,
                                   label=legend_plot[i]+text_n_data)

                        # plot axis
                        ax.set_xlim(0, params['plotDis']['x_max'])
                        ax.set_xticks(np.arange(0, params['plotDis']['x_max']+.1, step=0.4))
                        if i == con_nrs-1:
                            ax.set_xlabel(f'{foci_name} - {ref_name}\n 3D distance (μm)')
                        ax.set_ylim(y_min, y_max_foci)
                        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 0))
                        ax.yaxis.get_offset_text().set_visible(params['plt_sci_text'])
                        if params['legend'] == 1:
                            leg = ax.legend(bbox_to_anchor=(1, 0.5), loc='center left',
                                            fontsize='large', markerscale=2)
                            for lh in leg.legendHandles:
                                lh.set_alpha(1)

                    # plot threshold vertical -- line
                    for i, ax in enumerate(fig.axes):
                        ax.vlines(threshold, 0, y_max_foci, colors=ecolor, linestyles='--')
                        ax.locator_params(axis='y', nbins=3)

                    fig.supylabel(label, x=-0.08)
                    plt.savefig(f"{path_out}_{name_plot}_3D_{par}_scatter{params['file_type']}", bbox_inches='tight')
                    plt.close(fig)

                # --- Scat 2D distance vs foci param ---
                if params['plotDis']['plot_par_xD'] == '2D':
                    col_dis_RDM = f"{foci_name}-RDM 2D"  # 2D distances between foci and random nuclear spot

                    fig, axs = plt.subplots(figsize=(scalar * 2, height * con_nrs), ncols=2,
                                            nrows=con_nrs, sharex=True, sharey=True)

                    for i, (ax0, ax1) in enumerate(zip(fig.axes[::2], fig.axes[1::2])):
                        con_name = sort_values[i]

                        # plot data
                        data_plt = df_plot.loc[con_name, [col_dis, col_dis_RDM, par_foci]].dropna()
                        data_dis_ref = data_plt[col_dis].to_numpy()
                        data_dis_RDM = data_plt[col_dis_RDM].to_numpy()
                        data_int_foci = data_plt[par_foci].to_numpy()

                        ax0.scatter(data_dis_ref, data_int_foci, c=color_plot[i], alpha=alpha, s=(scalar+1)**2,
                                    label=legend_plot[i])
                        ax1.scatter(data_dis_RDM, data_int_foci, c=color_plot[i], alpha=alpha, s=(scalar+1)**2,
                                    label=legend_plot[i])

                        if params['legend'] == 1:
                            leg = ax1.legend(bbox_to_anchor=(1, 0.5), loc='center left',
                                             fontsize='large', markerscale=2)
                            for lh in leg.legendHandles:
                                lh.set_alpha(1)
                        if i == con_nrs-1:  # plot x axis label under lowest plot
                            ax0.set_xlabel(f'{foci_name} - {ref_name}\n 2D distance (μm)')
                            ax1.set_xlabel(f'{foci_name} - RDM\n 2D distance (μm)')

                        ax0.set_ylim(y_min, y_max_foci)
                        ax1.set_ylim(y_min, y_max_foci)

                    for i, ax in enumerate(fig.axes):
                        # plot threshold vertical -- line
                        ax.vlines(threshold, 0, y_max_foci, colors=ecolor, linestyles='--')
                        ax.set_xlim(0, params['plotDis']['x_max'])
                        ax.set_xticks(np.arange(0, params['plotDis']['x_max']+.1, step=0.4))
                        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 0))
                        ax.yaxis.get_offset_text().set_visible(params['plt_sci_text'])
                        ax.locator_params(axis='y', nbins=3)
                    fig.supylabel(f'{foci_name} {label}', x=0.04)
                    plt.savefig(f"{path_out}_{name_plot}_2D_{par}_scatter{params['file_type']}", bbox_inches='tight')
                    plt.close(fig)

                # 1) Scatterplot of ref par vs par of all foci
                fig, axs = plt.subplots(figsize=(scalar, height * con_nrs), ncols=1, nrows=con_nrs, sharex=True)
                for i, ax in enumerate(fig.axes):
                    con_name = sort_values[i]

                    # plot data
                    data_plt = df_plot.loc[con_name, [par_ref, par_foci]].dropna()
                    data_int_ref = data_plt[par_ref].to_numpy()
                    data_int_foci = data_plt[par_foci].to_numpy()
                    text_n_data = f"{len(data_int_ref)} spots)"
                    ax.scatter(data_int_ref, data_int_foci, c=color_plot[i], alpha=alpha, s=(scalar+1)**2,
                               label=legend_plot[i]+text_n_data)

                    # plot axis
                    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 0))
                    ax.yaxis.get_offset_text().set_visible(params['plt_sci_text'])
                    if params['legend'] == 1:
                        leg = ax.legend(bbox_to_anchor=(1, 0.5), loc='center left',
                                        fontsize='large', markerscale=2)
                        for lh in leg.legendHandles:
                            lh.set_alpha(1)
                    if i == 0:
                        ax.set_title(f'All {xD}\n', fontsize=font_size)
                    if i == con_nrs-1:
                        ax.set_xlabel(f'\n{ref_name} {label}')
                    ax.ticklabel_format(style='sci', axis='x', scilimits=(-2, 0))
                    ax.xaxis.get_offset_text().set_visible(params['plt_sci_text'])
                    ax.set_xlim(y_min, y_max_ref)

                    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 0))
                    ax.yaxis.get_offset_text().set_visible(params['plt_sci_text'])
                    ax.set_ylim(y_min, y_max_foci)
                    ax.locator_params(axis='y', nbins=3)

                fig.supylabel(f'{foci_name} {label}', x=-0.08)
                plt.savefig(f"{path_out}_{name_plot}_{xD}_{par}_scatter_all{params['file_type']}", bbox_inches='tight')
                plt.close(fig)

                # 2) Scatterplot of ref intensity vs intensity of overlapping foci
                fig, axs = plt.subplots(figsize=(scalar, height * con_nrs), ncols=1, nrows=con_nrs, sharex=True)

                for i, ax in enumerate(fig.axes):
                    con_name = sort_values[i]

                    # plot data
                    data_plt = df_plot.loc[con_name, ['file', 'cell', col_dis, par_ref, par_foci]].dropna()
                    data_plt = data_plt.query(f"`{col_dis}`<{threshold}")  # filter on rows with a distance below threshold
                    idx = data_plt.groupby(['file', 'cell'])[col_dis].transform(min) == data_plt[col_dis]  # nearest == True
                    data_plt = data_plt[idx]  # get rows which contain nearest

                    data_int_ref = data_plt[par_ref].to_numpy()
                    data_int_foci = data_plt[par_foci].to_numpy()
                    text_n_data = f"{len(data_int_ref)} spots)"
                    ax.scatter(data_int_ref, data_int_foci, c=color_plot[i], alpha=alpha, s=(scalar+1)**2,
                               label=legend_plot[i]+text_n_data)

                    # plot axis
                    if params['legend'] == 1:
                        leg = ax.legend(bbox_to_anchor=(1, 0.5), loc='center left',
                                        fontsize='large', markerscale=2)
                        for lh in leg.legendHandles:
                            lh.set_alpha(1)
                    if i == 0:
                        ax.set_title(f'Overlap + nearest {xD}\n', fontsize=font_size)
                    if i == con_nrs-1:
                        ax.set_xlabel(f'\n{ref_name} {label}')
                    ax.ticklabel_format(style='sci', axis='x', scilimits=(-2, 0))
                    ax.xaxis.get_offset_text().set_visible(params['plt_sci_text'])
                    ax.set_xlim(y_min, y_max_ref)

                    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 0))
                    ax.yaxis.get_offset_text().set_visible(params['plt_sci_text'])
                    ax.set_ylim(y_min, y_max_foci)
                    ax.locator_params(axis='y', nbins=3)

                fig.supylabel(f'{foci_name} {label}', x=-0.08)
                plt.savefig(f"{path_out}_{name_plot}_{xD}_{par}_scatter_nearest{params['file_type']}", bbox_inches='tight')
                plt.close(fig)

                # 3) Jitterplot foci intensity (dis < threshold) vs foci intensity (dis => threshold)
                fig, axs = plt.subplots(figsize=(scalar/2, height * con_nrs), ncols=1, nrows=con_nrs, sharex=True)
                for i, ax in enumerate(fig.axes):
                    con_name = sort_values[i]

                    # plot data
                    data_plt = df_plot.loc[con_name, ['file', 'cell', col_dis, par_foci]].dropna()
                    data_plt_below = data_plt.query(f"`{col_dis}`<{threshold}")[par_foci].to_numpy()
                    data_plt_above = data_plt.query(f"`{col_dis}`>={threshold}")[par_foci].to_numpy()

                    text = ''
                    if data_plt_below.any() and data_plt_above.any():
                        # Perform Mann-Whitney U test
                        statistic, pvalue = ss.mannwhitneyu(data_plt_below, data_plt_above)
                        text = f"Overlapping {foci_name.lower()} have {'a' if pvalue < 0.05 else 'no'} different {par} distribution \n" \
                               f"         (Mann-Whitney U test; p = {pvalue:.3g})\n\n"
                    sns.stripplot(data=[data_plt_below, data_plt_above], ax=ax, jitter=0.20, size=scalar+1,
                                  palette=[color_plot[i], 'grey'], alpha=0.5)
                    sns.boxplot(data=[data_plt_below, data_plt_above], ax=ax, color='lightgrey', showfliers=False)

                    text_n_data = f"{len(data_plt_below)} below / {len(data_plt_above)} above spots)"

                    # plot axis
                    if params['legend'] == 1:
                        ax.text(x=2, y=0.5*y_max_foci, s=text + legend_plot[i] + text_n_data, ha='left', va='center')
                    ax.set_ylim(y_min, y_max_foci)
                    ax.locator_params(axis='y', nbins=3)
                    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 0))
                    ax.yaxis.get_offset_text().set_visible(params['plt_sci_text'])

                    if i == con_nrs-1:
                        ax.set_xlabel(f'{xD} distance of {threshold} μm')
                        ax.set_xticklabels([f'closer', 'further'])

                fig.supylabel(f'{foci_name} {label}', x=-0.26)
                plt.savefig(f"{path_out}_{name_plot}_{xD}_{par}_foci_vs_overlap{params['file_type']}", bbox_inches='tight')
                plt.close(fig)

                # 4) Jitterplot ref intensity (dis < threshold) vs ref intensity (dis => threshold)
                fig, axs = plt.subplots(figsize=(scalar/2, height * con_nrs), ncols=1, nrows=con_nrs, sharex=True)
                for i, ax in enumerate(fig.axes):
                    con_name = sort_values[i]

                    # plot data
                    data_plt = df_plot.loc[con_name, ['file', 'cell', col_dis, par_ref]].dropna()
                    idx = data_plt.groupby(['file', 'cell'])[col_dis].transform(min) == data_plt[
                        col_dis]  # select rows with nearest col_dis per cell to plot each ref only once
                    data_plt = data_plt[idx]  # get rows which contain nearest
                    data_plt_below = data_plt.query(f"`{col_dis}`<{threshold}")[par_ref].to_numpy()
                    data_plt_above = data_plt.query(f"`{col_dis}`>={threshold}")[par_ref].to_numpy()

                    text = ''
                    if data_plt_below.any() and data_plt_above.any():
                        # Perform Mann-Whitney U test
                        statistic, pvalue = ss.mannwhitneyu(data_plt_below, data_plt_above)
                        text = f"Overlapping {ref_name} have {'a' if pvalue < 0.05 else 'no'} different {par} distribution \n" \
                               f"         (Mann-Whitney U test; p = {pvalue:.3g})\n\n"
                    sns.stripplot(data=[data_plt_below, data_plt_above], ax=ax, jitter=0.20, size=scalar+1,
                                  palette=['#f768a1', 'grey'], alpha=0.5)
                    sns.boxplot(data=[data_plt_below, data_plt_above], ax=ax, color='lightgrey', showfliers=False)

                    text_n_data = f"{len(data_plt_below)} below / {len(data_plt_above)} above spots)"

                    # plot axis
                    if params['legend'] == 1:
                        ax.text(x=2, y=0.5*y_max_ref, s=text + legend_plot[i] + text_n_data, ha='left', va='center')
                    ax.set_ylim(y_min, y_max_ref)
                    ax.locator_params(axis='y', nbins=3)
                    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 0))
                    ax.yaxis.get_offset_text().set_visible(params['plt_sci_text'])

                    if i == con_nrs-1:
                        ax.set_xlabel(f'{xD} distance of {threshold} μm')
                        ax.set_xticklabels([f'closer', 'further'])

                fig.supylabel(f'{ref_name} {label}', x=-0.26)
                plt.savefig(f"{path_out}_{name_plot}_{xD}_{par}_ref_vs_overlap{params['file_type']}", bbox_inches='tight')
                plt.close(fig)


def weights(n_data, n_cells):
    """
    calculate weight of each data point, normalized for cell number and data point number of replicate
    n_data  = array of data point number of replicates
    n-cells = array of cell number of replicates
    NOTE: n_data needs to be in same order as n_cells
    """
    w = sum([d * (c / d / sum(n_cells),) for c, d in zip(n_cells, n_data)], ())
    return w


def bootstrap(data, w, bins, n=10000, norm=True):
    """
    calculate sd of bootstrapped means which is equal to the standard error of the mean (sem)
    data = array-like data points
    w    = array-like weight for each data point
    bins = array-like bin edges
    n    = number of recursions
    norm = normalization to number of data points provided
    """
    i = np.random.choice(range(len(data)), (int(n), len(data)), True, np.array(w) / sum(w))
    h = [np.histogram(d, bins)[0] for d in data[i]]
    bsem = np.std(h, 0)  # bootstrapped sem
    hist = np.histogram(data, bins)[0]  # count hist of data
    if norm:
        hist_norm = hist/len(data)  # normalized hist of data using provided bins
        bsem_norm = np.divide(hist_norm, hist, out=np.zeros_like(hist_norm), where=hist != 0) * bsem  # normalized bootstrapped sem corresponding to hist_norm
        return hist_norm, bsem_norm
    elif not norm:
        return hist, bsem
    elif type(norm) == int or type(norm) == float:  # norm = value to normalize counts to
        hist_norm = hist/norm  # normalized hist of data using provided bins
        bsem_norm = np.divide(hist_norm, hist, out=np.zeros_like(hist_norm), where=hist != 0) * bsem  # normalized bootstrapped sem corresponding to hist_norm
        return hist_norm, bsem_norm


def foci_figs_pipeline(parameter_file):
    params = misc.getParams(parameter_file, __file__.replace('.py', '_parameters_template.yml'),
                            ('path_in', 'path_out'))

    os.makedirs(params['path_out'], exist_ok=True)

    ld_rep_paths = get_rep_paths(params)
    name_plot = plot_name(params, ld_rep_paths)

    if params['plotNum'].get('plot'):
        plot_num(params, ld_rep_paths, name_plot)

    if params['plotPar'].get('plot'):
        plot_par(params, ld_rep_paths, name_plot)

    if params['plotNrPar'].get('plot'):
        plot_nr_par(params, ld_rep_paths, name_plot)

    if params['plotDis'].get('plot'):
        plot_dis(params, ld_rep_paths, name_plot)


def main():
    misc.ipy_debug()
    if len(sys.argv) < 2:
        if os.path.exists('foci_plot_figures_combined_parameters.yml'):
            parameter_file = 'foci_plot_figures_combined_parameters.yml'
        elif os.path.exists('foci_plot_figures_combined_parameters.yml'):
            parameter_file = 'foci_plot_figures_combined_parameters.yml'
        else:
            raise FileNotFoundError('Could not find the parameter file.')
        print('Using ' + parameter_file)
    else:
        parameter_file = sys.argv[1]

    tm = time()
    foci_figs_pipeline(parameter_file)

    print('------------------------------------------------')
    print(misc.color('Pipeline finished, took {} seconds.'.format(time()-tm), 'g:b'))


if __name__ == '__main__':
    main()
    imr.kill_vm()
