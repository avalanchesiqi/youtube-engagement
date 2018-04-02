#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to plot fitting error from power law, linear and constant fitting.

Usage: python plot_fig6_fit_results.py
Time: ~1M
"""

import time, datetime
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to plot fitting error from power law, linear and constant fitting...')
    start_time = time.time()

    mae_powerlaw = []
    mae_linear = []
    mae_constant = []

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    input_loc = '../temporal_analysis/sliding_fitting_results.csv'

    with open(input_loc, 'r') as fin:
        fin.readline()
        for line in fin:
            vid, error_powerlaw, error_linear, error_constant = line.rstrip().split(',')
            mae_powerlaw.append(float(error_powerlaw))
            mae_linear.append(float(error_linear))
            mae_constant.append(float(error_constant))

    # == == == == == == == == Part 3: Plot fitting results == == == == == == == == #
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)

    mae_matrix = [mae_powerlaw, mae_linear, mae_constant]
    power_box = [mae_powerlaw, [], []]
    linear_box = [[], mae_linear, []]
    constant_box = [[], [], mae_constant]
    combined_box = [power_box, linear_box, constant_box]

    label_array = ['Power-law\n$at^b+c$', 'Linear\n$wt+b$', 'Constant\n$c$']
    boxplots = [ax1.boxplot(box, labels=label_array, showfliers=False, showmeans=True,
                            widths=0.6) for box in combined_box]

    box_colors = ['b', 'r', 'g']
    n_box = len(combined_box)
    for bplot, bcolor in zip(boxplots, box_colors[:n_box]):
        plt.setp(bplot['boxes'], color=bcolor)
        plt.setp(bplot['whiskers'], color=bcolor)
        plt.setp(bplot['caps'], color=bcolor)

    numBoxes = 2 * n_box
    medians = list(range(numBoxes))
    for i, bplot, bcolor in zip([0, 1, 2], boxplots, box_colors[:n_box]):
        # Now draw the median lines back over what we just filled in
        med = bplot['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            plt.plot(medianX, medianY, color=bcolor, lw=1.5, zorder=30)
            medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment in the center of each box
        plt.plot([np.average(med.get_xdata())], [np.average(mae_matrix[i])],
                 color=bcolor,
                 marker='s', markeredgecolor=bcolor,
                 zorder=30)

    means = [np.mean(x) for x in [mae_powerlaw, mae_linear, mae_constant]]
    means_labels = ['{0:.4f}'.format(s) for s in means]
    pos = range(len(means))
    for tick, label in zip(pos, ax1.get_xticklabels()):
        ax1.text(pos[tick] + 1, means[tick] + 0.005, means_labels[tick], horizontalalignment='center', size=20, color='k')
    ax1.set_ylabel('mean absolute error', fontsize=18)
    ax1.set_yticks([0, 0.05, 0.10, 0.15])
    ax1.tick_params(axis='x', which='major', labelsize=18)
    ax1.tick_params(axis='y', which='major', labelsize=18)
    ax1.set_title('(b)', fontsize=24)

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    plt.tight_layout()
    plt.savefig('../images/fig6_fit_results.pdf', bbox_inches='tight')
    plt.show()
