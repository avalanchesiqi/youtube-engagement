#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to plot Figure 3, engagement map of watch time and watch percentage
# Fields:
# 'id', 'publish', 'duration', 'definition', 'category', 'detect_lang', 'channel', 'topics',
# 'view@30', 'watch@30', 'wp@30', 'days', 'daily_view', 'daily_watch'

Input: ../../production_data/new_tweeted_dataset_norm/
Usage: python plot_fig3_emaps.py
Time: ~10M
"""

from __future__ import print_function, division
import os, sys, time, datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
from collections import defaultdict
from scipy.stats import gaussian_kde
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter


def exponent(x, pos):
    'The two args are the value and tick position'
    return '%1.0f' % (10 ** x)


def get_engagement_stats_from_file(filepath):
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            _, _, duration, dump = line.rstrip().split('\t', 3)
            _, _, _, _, _, view, _, wp30, _ = dump.split('\t', 8)
            duration = int(duration)
            wp30 = float(wp30)
            duration_engagement_tuple.append((duration, wp30, np.log10(duration * wp30)))
            duration_cnt_dict[duration] += 1


def plot_contour(x_axis_value, color='r', fsize=14, title=False):
    """Plot relative engagement contour within one duration bin."""
    target_bin = wp_bin_matrix[np.sum(x_axis < x_axis_value)]
    ax1.plot([x_axis_value, x_axis_value], [np.percentile(target_bin, 0.5), np.percentile(target_bin, 99.5)], c=color,
             zorder=20)
    for t in xrange(10, 95, 10):
        ax1.plot([x_axis_value - 0.04, x_axis_value + 0.04],
                 [np.percentile(target_bin, t), np.percentile(target_bin, t)], c=color, zorder=20)
        if t % 20 == 0:
            ax1.text(x_axis_value + 0.1, np.percentile(target_bin, t), '{0}%'.format(int(t)), fontsize=fsize,
                     verticalalignment='center', zorder=20)
    for t in [0.5, 99.5]:
        ax1.plot([x_axis_value - 0.04, x_axis_value + 0.04],
                 [np.percentile(target_bin, t), np.percentile(target_bin, t)], c=color, zorder=20)
    if title:
        ax1.text(x_axis_value, np.percentile(target_bin, 99.5) + 0.02, r'$\bar \eta_{30}$', color='k', fontsize=28,
                 horizontalalignment='center', zorder=20)


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    print('>>> Start to extract engagement map and plot...')
    start_time = time.time()

    bin_number = 1000
    duration_engagement_tuple = []
    duration_cnt_dict = defaultdict(int)

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    input_loc = '../../production_data/new_tweeted_dataset'

    if os.path.isdir(input_loc):
        for subdir, _, files in os.walk(input_loc):
            for f in files:
                get_engagement_stats_from_file(os.path.join(subdir, f))
                print('>>> Loading data: {0} done!'.format(f))
    else:
        get_engagement_stats_from_file(input_loc)
    print('>>> Finish loading all data!')

    # == == == == == == == == Part 3: Build wp matrix based on duration splits == == == == == == == == #
    # sort by duration in ascent order
    sorted_duration_engagement_tuple = sorted(duration_engagement_tuple, key=lambda x: x[0])

    # get duration split point
    duration_list = sorted(duration_cnt_dict.keys())
    even_split_points = list(np.linspace(1, 5, bin_number))

    # put videos in correct bins
    wp_bin_matrix = []
    wp_individual_bin = []
    wt_bin_matrix = []
    wt_individual_bin = []
    bin_idx = 0
    duration_splits = []
    # put dur-wp tuple in the correct bin
    for duration, wp30, wt30 in sorted_duration_engagement_tuple:
        if np.log10(duration) > even_split_points[bin_idx]:
            bin_idx += 1
            # must contains at least 50 videos
            if len(wp_individual_bin) >= 50:
                wp_bin_matrix.append(wp_individual_bin)
                wp_individual_bin = []
                wt_bin_matrix.append(wt_individual_bin)
                wt_individual_bin = []
                duration_splits.append(duration - 1)
        wp_individual_bin.append(wp30)
        wt_individual_bin.append(wt30)
    if len(wp_individual_bin) > 0:
        wp_bin_matrix.append(np.array(wp_individual_bin))
        wt_bin_matrix.append(np.array(wt_individual_bin))

    # == == == == == == == == Part 4: Plot engagement map == == == == == == == == #
    cornflower_blue = '#6495ed'
    exp_formatter = FuncFormatter(exponent)
    to_plot_engagement_map = True
    if to_plot_engagement_map:
        gs = gridspec.GridSpec(2, 2, width_ratios=[8, 1], height_ratios=[1, 8])
        fig = plt.figure(figsize=(9, 9.5))
        ax1 = plt.subplot(gs[1, 0])

        x_axis = [np.log10(x) for x in duration_splits]
        for t in np.arange(5, 50, 5):
            ax1.fill_between(x_axis, [np.percentile(x, 50 - t) for x in wp_bin_matrix[:-1]],
                             [np.percentile(x, 55 - t) for x in wp_bin_matrix[:-1]],
                             facecolor=cornflower_blue, alpha=(100 - 2 * t) / 100, lw=0)
            ax1.fill_between(x_axis, [np.percentile(x, 45 + t) for x in wp_bin_matrix[:-1]],
                             [np.percentile(x, 50 + t) for x in wp_bin_matrix[:-1]],
                             facecolor=cornflower_blue, alpha=(100 - 2 * t) / 100, lw=0)

        for t in [10, 30, 70, 90]:
            ax1.plot(x_axis, [np.percentile(x, t) for x in wp_bin_matrix[:-1]], color=cornflower_blue, alpha=0.8,
                     zorder=15)
        ax1.plot(x_axis, [np.percentile(x, 50) for x in wp_bin_matrix[:-1]], color=cornflower_blue, alpha=1, zorder=15)

        ax1.set_xticks([1, 2, 3, 4, 5])
        ax1.xaxis.set_major_formatter(exp_formatter)
        ax1.set_xlim([1, 5])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('video duration (sec) ' + r'$D$', fontsize=24)
        ax1.set_ylabel('average watch percentage ' + r'$\bar \mu_{30}$', fontsize=24)
        ax1.tick_params(axis='both', which='major', labelsize=20)

        # KDE for top marginal
        df_x = [np.log10(x[0]) for x in duration_engagement_tuple]
        kde_x = gaussian_kde(df_x)
        # KDE for right marginal
        df_y = [x[1] for x in duration_engagement_tuple]
        kde_y = gaussian_kde(df_y)

        xmin, xmax = 1, 5
        ymin, ymax = 0, 1
        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)

        # Create Y-marginal (right)
        max_xlim = 1.2 * kde_y(y).max()
        axr = plt.subplot(gs[1, 1], xticks=[], yticks=[], frameon=False, xlim=(0, max_xlim), ylim=(ymin, ymax))
        axr.plot(kde_y(y), y, color=cornflower_blue)

        # Create X-marginal (top)
        max_ylim = 1.2 * kde_x(x).max()
        axt = plt.subplot(gs[0, 0], xticks=[], yticks=[], frameon=False, xlim=(xmin, xmax), ylim=(0, max_ylim))
        axt.plot(x, kde_x(x), color=cornflower_blue)

        plot_examples = True
        if plot_examples:
            # d_8ao3o5ohU, Black Belt Kid Vs. White Belt Adults, 6309812
            quality_short = (287, 0.7022605, '$\mathregular{V_{1}}$: d_8ao3o5ohU')
            # akuyBBIbOso, Learn Colors with Squishy Mesh Balls for Toddlers Kids and Children - Surprise Eggs for Babies, 6449735
            junk_short = (306, 0.2066883, '$\mathregular{V_{2}}$: akuyBBIbOso')
            # WH7llf2vaKQ, Joe Rogan Experience - Fight Companion - August 6, 2016, 490585
            quality_long = (13779, 0.1900219, '$\mathregular{V_{3}}$: WH7llf2vaKQ')

            points = [quality_short, junk_short, quality_long]
            for point in points:
                ax1.scatter(np.log10(point[0]), point[1], s=30, facecolor='#ff4500', edgecolor='k', lw=1, zorder=25)
                ax1.text(np.log10(point[0]), point[1] + 0.02, point[2],
                         horizontalalignment='center', size=20, color='k', zorder=25)

            plot_contour((np.log10(287) + np.log10(306)) / 2, color='k', fsize=18, title=True)
            plot_contour(np.log10(13779), color='k', fsize=14)

        axt.set_title('(b)', fontsize=32)
        plt.subplots_adjust(left=0.13, bottom=0.08, right=0.99, top=0.96, wspace=0.03, hspace=0.03)
        plt.savefig('../images/fig3_emap_wp.pdf', bbox_inches='tight')
        plt.show()

    # == == == == == == == == Part 6: Plot duration watch time map == == == == == == == == #
    sea_green = '#2e8b57'
    to_plot_watch_time = True
    if to_plot_watch_time:
        gs = gridspec.GridSpec(2, 2, width_ratios=[8, 1], height_ratios=[1, 8])
        fig = plt.figure(figsize=(9, 9.5))
        ax1 = plt.subplot(gs[1, 0])

        x_axis = [np.log10(x) for x in duration_splits]
        for t in np.arange(5, 50, 5):
            ax1.fill_between(x_axis, [np.percentile(x, 50 - t) for x in wt_bin_matrix[:-1]],
                             [np.percentile(x, 55 - t) for x in wt_bin_matrix[:-1]],
                             facecolor=sea_green, alpha=(100 - 2 * t) / 100, lw=0)
            ax1.fill_between(x_axis, [np.percentile(x, 45 + t) for x in wt_bin_matrix[:-1]],
                             [np.percentile(x, 50 + t) for x in wt_bin_matrix[:-1]],
                             facecolor=sea_green, alpha=(100 - 2 * t) / 100, lw=0)

        for t in [10, 30, 70, 90]:
            ax1.plot(x_axis, [np.percentile(x, t) for x in wt_bin_matrix[:-1]], color=sea_green, alpha=0.8, zorder=15)
        ax1.plot(x_axis, [np.percentile(x, 50) for x in wt_bin_matrix[:-1]], color=sea_green, alpha=1, zorder=15)

        ax1.set_xticks([1, 2, 3, 4, 5])
        ax1.xaxis.set_major_formatter(exp_formatter)
        ax1.yaxis.set_major_formatter(exp_formatter)
        ax1.set_xlim([1, 5])
        ax1.set_ylim([1, 5])
        ax1.set_xlabel('video duration (sec) ' + r'$D$', fontsize=24)
        ax1.set_ylabel('average watch time (sec) ' + r'$\omega_{30}$', fontsize=24)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        for label in ax1.get_yticklabels()[1::2]:
            label.set_visible(False)
        plt.setp(ax1.yaxis.get_majorticklabels(), rotation=90)

        # KDE for top marginal
        df_x = [np.log10(x[0]) for x in duration_engagement_tuple]
        kde_x = gaussian_kde(df_x)
        # KDE for right marginal
        df_y = [x[2] for x in duration_engagement_tuple]
        kde_y = gaussian_kde(df_y)

        xmin, xmax = 1, 5
        ymin, ymax = 1, 5
        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)

        # Create Y-marginal (right)
        max_xlim = 1.2 * kde_y(y).max()
        axr = plt.subplot(gs[1, 1], xticks=[], yticks=[], frameon=False, xlim=(0, max_xlim), ylim=(ymin, ymax))
        axr.plot(kde_y(y), y, color=sea_green)

        # Create X-marginal (top)
        max_ylim = 1.2 * kde_x(x).max()
        axt = plt.subplot(gs[0, 0], xticks=[], yticks=[], frameon=False, xlim=(xmin, xmax), ylim=(0, max_ylim))
        axt.plot(x, kde_x(x), color=sea_green)

        plot_examples = True
        if plot_examples:
            # d_8ao3o5ohU, Black Belt Kid Vs. White Belt Adults, 6309812
            quality_short = (287, 0.7022605 * 287, '$\mathregular{V_{1}}$: d_8ao3o5ohU')
            # akuyBBIbOso, Learn Colors with Squishy Mesh Balls for Toddlers Kids and Children - Surprise Eggs for Babies, 6449735
            junk_short = (306, 0.2066883 * 306, '$\mathregular{V_{2}}$: akuyBBIbOso')
            # WH7llf2vaKQ, Joe Rogan Experience - Fight Companion - August 6, 2016, 490585
            quality_long = (13779, 0.1900219 * 13779, '$\mathregular{V_{3}}$: WH7llf2vaKQ')

            points = [quality_short, junk_short, quality_long]
            for point in points:
                ax1.scatter(np.log10(point[0]), np.log10(point[1]), s=30, facecolor='#ff4500', edgecolor='k', lw=1, zorder=25)
                ax1.text(np.log10(point[0]), np.log10(point[1]) + 0.02, point[2],
                         horizontalalignment='center', size=20, color='k', zorder=25)

        axt.set_title('(a)', fontsize=32)
        plt.subplots_adjust(left=0.13, bottom=0.08, right=0.99, top=0.96, wspace=0.03, hspace=0.03)
        plt.savefig('../images/fig3_emap_wt.pdf', bbox_inches='tight')
        plt.show()

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])
