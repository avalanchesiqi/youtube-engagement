#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Scripts to plot Figure 4, compare quality music and random music engagement map."""

from __future__ import print_function, division
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from collections import defaultdict
from scipy.stats import gaussian_kde
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def get_duration_wp_from_file(filepath, duration_wp_tuple, duration_stats_dict):
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            _, _, duration, dump = line.rstrip().split('\t', 3)
            _, _, _, _, _, view30, _, wp30, _ = dump.split('\t', 8)
            duration = int(duration)
            wp30 = float(wp30)
            if int(view30) >= 100:
                duration_wp_tuple.append((duration, wp30))
                duration_stats_dict[duration] += 1


def remove_bad_bins(x_axis, bin_matrix, min_bin=25):
    x_axis = x_axis[:len(bin_matrix)]
    bad_bin_idx = []
    for idx, bin in enumerate(bin_matrix):
        if len(bin) < min_bin:
            bad_bin_idx.append(idx)
    for idx in bad_bin_idx[::-1]:
        x_axis.pop(idx)
        bin_matrix.pop(idx)
    return x_axis, bin_matrix


def loading_data(input_loc, bin_number, min_bin):
    # setting parameters
    duration_wp_tuple = []
    duration_stats_dict = defaultdict(int)

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    if os.path.isdir(input_loc):
        for subdir, _, files in os.walk(input_loc):
            for f in files:
                # get tweeted music videos
                if f.startswith('10'):
                    get_duration_wp_from_file(os.path.join(subdir, f), duration_wp_tuple, duration_stats_dict)
                    print('>>> Loading data: {0} done!'.format(f))
    else:
        get_duration_wp_from_file(input_loc, duration_wp_tuple, duration_stats_dict)
    print('>>> Finish loading all data!')

    # sort by duration in ascent order
    sorted_duration_wp_tuple = sorted(duration_wp_tuple, key=lambda x: x[0])

    # get duration split point
    x_axis = list(np.linspace(xmin, xmax, bin_number))

    # put videos in correct bins
    bin_matrix = []
    bin_list = []
    bin_idx = 0
    # put dur-wp tuple in the correct bin
    for item in sorted_duration_wp_tuple:
        if np.log10(item[0]) > x_axis[bin_idx]:
            bin_matrix.append(bin_list)
            bin_idx += 1
            bin_list = []
        bin_list.append(item[1])
    if len(bin_list) > 0:
        bin_matrix.append(bin_list)
    bin_matrix = [np.array(x) for x in bin_matrix]

    x_axis, bin_matrix = remove_bad_bins(x_axis, bin_matrix, min_bin)

    # sanity check
    to_check = True
    if to_check:
        print('videos in each bin')
        # for i in xrange(len(x_axis)):
        #     print('duration split point: {0}; number of videos in bin: {1}'.format(x_axis[i], len(bin_matrix[i])))
        print('num of bins: {0}'.format(len(x_axis)))
    return x_axis, bin_matrix, duration_wp_tuple


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    xmin, xmax = 1, 5
    ymin, ymax = 0, 1
    cornflower_blue = '#6495ed'
    tomato = '#ff6347'

    plot_tweeted = True
    if plot_tweeted:
        input_loc = '../../production_data/new_tweeted_dataset'
        tweeted_x_axis, tweeted_bin_matrix, tweeted_tuple = loading_data(input_loc, bin_number=150, min_bin=50)

    plot_quality = True
    if plot_quality:
        input_loc = '../../production_data/quality_dataset/vevo.txt'
        quality_x_axis, quality_bin_matrix, quality_tuple = loading_data(input_loc, bin_number=50, min_bin=25)

        input_loc2 = '../../production_data/quality_dataset/billboard_2016.txt'
        _, _, quality_tuple2 = loading_data(input_loc2, bin_number=50, min_bin=25)

    # plot wp~dur distribution
    to_plot = True
    if to_plot:
        gs = gridspec.GridSpec(2, 2, width_ratios=[8, 1], height_ratios=[1, 8])
        fig = plt.figure(figsize=(9, 9.5))
        ax1 = plt.subplot(gs[1, 0])

        for t in np.arange(5, 50, 5):
            ax1.fill_between(tweeted_x_axis, [np.percentile(x, 50-t) for x in tweeted_bin_matrix],
                             [np.percentile(x, 55-t) for x in tweeted_bin_matrix], facecolor=cornflower_blue, alpha=(100-2*t)/100, lw=0, zorder=1)
            ax1.fill_between(tweeted_x_axis, [np.percentile(x, 45+t) for x in tweeted_bin_matrix],
                             [np.percentile(x, 50+t) for x in tweeted_bin_matrix], facecolor=cornflower_blue, alpha=(100-2*t)/100, lw=0, zorder=1)
        for t in [10, 30, 70, 90]:
            ax1.plot(tweeted_x_axis, [np.percentile(x, t) for x in tweeted_bin_matrix], color=cornflower_blue, alpha=1, zorder=5)
        ax1.plot(tweeted_x_axis, [np.percentile(x, 50) for x in tweeted_bin_matrix], color='k', alpha=0.5, zorder=20, lw=2)

        for t in np.arange(5, 30, 5):
            ax1.fill_between(quality_x_axis, [np.percentile(x, 50-t) for x in quality_bin_matrix],
                             [np.percentile(x, 55-t) for x in quality_bin_matrix], facecolor=tomato, alpha=0.8*(100-2*t)/100, lw=0, zorder=10)
            ax1.fill_between(quality_x_axis, [np.percentile(x, 45+t) for x in quality_bin_matrix],
                             [np.percentile(x, 50+t) for x in quality_bin_matrix], facecolor=tomato, alpha=0.8*(100-2*t)/100, lw=0, zorder=10)
        for t in [30, 70]:
            ax1.plot(quality_x_axis, [np.percentile(x, t) for x in quality_bin_matrix], color=tomato, alpha=1, zorder=15, linestyle='-')
        ax1.plot(quality_x_axis, [np.percentile(x, 50) for x in quality_bin_matrix], color=tomato, alpha=1, zorder=20, lw=2, linestyle='-')
        ax1.plot(quality_x_axis, [np.percentile(x, 50) for x in quality_bin_matrix], color='k', alpha=0.5, zorder=20, lw=2, linestyle='--')

        ax1.scatter([np.log10(x[0]) for x in quality_tuple2], [x[1] for x in quality_tuple2], marker='x', c='k', zorder=30)

        def exponent(x, pos):
            'The two args are the value and tick position'
            return '%1.0f' % (10 ** x)

        ax1.set_xticks([1, 2, 3, 4, 5])
        x_formatter = FuncFormatter(exponent)
        ax1.xaxis.set_major_formatter(x_formatter)
        ax1.set_xlim([xmin, xmax])
        ax1.set_ylim([ymin, ymax])
        ax1.set_xlabel('video duration (sec) '+r'$D$', fontsize=24)
        ax1.set_ylabel('average watch percentage '+r'$\bar \mu_{30}$', fontsize=24)
        ax1.tick_params(axis='both', which='major', labelsize=20)

        ax1.legend([plt.Rectangle((0, 0), 1, 1, fc=cornflower_blue), plt.Rectangle((0, 0), 1, 1, fc=tomato),
                    plt.Rectangle((0, 0), 1, 1, fc='k')],
                   ['Tweeted Music', 'Vevo', 'Billboard'], loc='upper right', fontsize=24, frameon=False)

        tweeted_df_x = [np.log10(x[0]) for x in tweeted_tuple]
        tweeted_df_y = [x[1] for x in tweeted_tuple]
        # KDE for top marginal
        tweeted_kde_x = gaussian_kde(tweeted_df_x)
        # KDE for right marginal
        tweeted_kde_y = gaussian_kde(tweeted_df_y)

        quality_df_x = [np.log10(x[0]) for x in quality_tuple]
        quality_df_y = [x[1] for x in quality_tuple]
        # KDE for top marginal
        quality_kde_x = gaussian_kde(quality_df_x)
        # KDE for right marginal
        quality_kde_y = gaussian_kde(quality_df_y)

        quality_df_x2 = [np.log10(x[0]) for x in quality_tuple2]
        quality_df_y2 = [x[1] for x in quality_tuple2]
        # KDE for top marginal
        quality_kde_x2 = gaussian_kde(quality_df_x2)
        # KDE for right marginal
        quality_kde_y2 = gaussian_kde(quality_df_y2)

        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)

        # Create Y-marginal (right)
        max_xlim = 1.2 * max([tweeted_kde_y(y).max(), quality_kde_y(y).max(), quality_kde_y2(y).max()])
        axr = plt.subplot(gs[1, 1], xticks=[], yticks=[], frameon=False, xlim=(0, max_xlim), ylim=(ymin, ymax))
        axr.plot(tweeted_kde_y(y), y, color=cornflower_blue)
        axr.plot(quality_kde_y(y), y, color=tomato)
        axr.plot(quality_kde_y2(y), y, color='k')

        # Create X-marginal (top)
        max_ylim = 1.2 * max([tweeted_kde_x(x).max(), quality_kde_x(x).max(), quality_kde_x2(x).max()])
        axt = plt.subplot(gs[0, 0], xticks=[], yticks=[], frameon=False, xlim=(xmin, xmax), ylim=(0, max_ylim))
        axt.plot(x, tweeted_kde_x(x), color=cornflower_blue)
        axt.plot(x, quality_kde_x(x), color=tomato)
        axt.plot(x, quality_kde_x2(x), color='k')

        axt.set_title('(a)', fontsize=32)
        plt.subplots_adjust(left=0.13, bottom=0.08, right=0.99, top=0.96, wspace=0.03, hspace=0.03)
        plt.savefig('../images/fig4_music_emap.pdf', bbox_inches='tight')
        plt.show()
