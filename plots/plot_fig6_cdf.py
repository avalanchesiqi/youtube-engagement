#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script to plot Figure 6, CDF of temporal change in relative engagement.

Usage: python plot_fig6_cdf.py
Time: ~3M
"""

import os, sys, platform
import numpy as np

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import read_as_int_array, read_as_float_array

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to plot cumulative relative engagement temporal change...')
    timer = Timer()
    timer.start()

    list1 = []  # day7 vs day14
    list2 = []  # day7 vs day30

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    with open('../temporal_analysis/cum_engagement_dynamics.csv', 'r') as fin:
        for line in fin:
            vid, days, _, re_list = line.rstrip().split('\t')
            days = read_as_int_array(days, delimiter=',')
            re_list = read_as_float_array(re_list, delimiter=',')
            if 6 in days:
                list1.append(re_list[days == 13] - re_list[days == 6])
                list2.append(re_list[days == 29] - re_list[days == 6])
    list1 = np.array(list1)
    list2 = np.array(list2)

    # == == == == == == == == Part 3: Plot relative engagement change == == == == == == == == #
    to_plot = True
    if to_plot:
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)

        n = len(list1)
        print('>>> Number of videos has records of relative engagement at day 7: ', n)
        # split into 60 bins, change less than 0.1, 40th bin, change less than -0.1, 20th bin
        start = -0.30
        end = 0.31
        step = 0.01
        x_axis = np.arange(start, end, step)
        y1 = [np.count_nonzero(list1 <= i) / n for i in x_axis]
        y2 = [np.count_nonzero(list2 <= i) / n for i in x_axis]

        ax1.plot(x_axis, y1, 'b--', lw=2, label=r'$\bar \eta_{14}-\bar \eta_{7}$')
        ax1.plot(x_axis, y2, 'r', lw=2, label=r'$\bar \eta_{30}-\bar \eta_{7}$')

        lower_y = np.count_nonzero(list2 <= -0.1) / n
        upper_y = np.count_nonzero(list2 < 0.1) / n
        ax1.scatter(-0.1, lower_y, c='k', s=30, zorder=30)
        ax1.plot([-0.1, 0.1], [lower_y, lower_y], 'k--', lw=1)
        ax1.scatter(0.1, upper_y, c='k', s=30, zorder=30)
        ax1.annotate('',
                     xy=(0.1, lower_y), xycoords='data',
                     xytext=(0.1, (lower_y + upper_y) / 2 - 0.05), textcoords='data',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        ax1.annotate('',
                     xy=(0.1, upper_y), xycoords='data',
                     xytext=(0.1, (lower_y + upper_y) / 2 + 0.12), textcoords='data',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        ax1.text(0.03, (lower_y + upper_y)/2, r'$F(0.1)-F(-0.1)=' + '{0:.1f}\%$'.format(100 * (upper_y - lower_y)), size=18)

        print('>>> {0:.2f}% videos increase less than 0.1.'.format(upper_y*100))
        print('>>> {0:.2f}% videos decrease more than -0.1.'.format(lower_y*100))

        ax1.set_xlabel(r'relative engagement change $\bar \eta_{t_2}-\bar \eta_{t_1}$', fontsize=18)
        ax1.set_ylabel('CDF', fontsize=18)
        ax1.set_xlim([-0.3, 0.3])
        ax1.set_ylim([0, 1])
        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.set_xticks([-0.3, -0.1, 0.1, 0.3])
        ax1.legend(loc='upper left', frameon=False, handlelength=1, fontsize=20)
        ax1.set_title('(a)', fontsize=24)

        timer.stop()

        plt.tight_layout()
        plt.savefig('../images/fig6_cum_engagement_change.pdf', bbox_inches='tight')
        if not platform.system() == 'Linux':
            plt.show()
