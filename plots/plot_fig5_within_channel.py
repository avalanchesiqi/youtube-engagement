#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script to plot Figure 5, compare engagement metrics within the same channel.
UCy-taFzDCV-XcpTBa3pF68w: PBABowling

Usage: python plot_fig5_within_channel.py
Time: ~1M
"""

import os, sys, time, datetime, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def exponent(x, pos):
    """ The two args are the value and tick position. """
    return '{0:.0f}'.format(x)


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to plot engagement metrics within the same channel...')
    start_time = time.time()

    channel_id = 'UCy-taFzDCV-XcpTBa3pF68w'
    channel_title = 'PBABowling'

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    channel_view_train_path = '../engagement_prediction/train_data_channel_view/UC{0}.p'.format(channel_id[2])
    channel_view_test_path = '../engagement_prediction/test_data_channel_view/UC{0}.p'.format(channel_id[2])
    if not os.path.exists(channel_view_train_path):
        print('>>> No train data in channel view found! Run construct_channel_view_dataset.py first!')
        sys.exit(1)
    if not os.path.exists(channel_view_test_path):
        print('>>> No test data in channel view found! Run construct_channel_view_dataset.py first!')
        sys.exit(1)

    channel_train_videos = pickle.load(open(channel_view_train_path, 'rb'))[channel_id]
    channel_test_videos = pickle.load(open(channel_view_test_path, 'rb'))[channel_id]

    channel_duration_list = []
    channel_watch_percentage_list = []
    channel_relative_engagement_list = []

    for channel_videos in [channel_train_videos, channel_test_videos]:
        for video in channel_videos:
            _, _, duration, tails = video.split('\t', 3)
            _, wp30, re30, _, _, _ = tails.rsplit('\t', 5)
            channel_duration_list.append(int(duration))
            channel_watch_percentage_list.append(float(wp30))
            channel_relative_engagement_list.append(float(re30))

    channel_duration_list = np.array(channel_duration_list)
    channel_watch_percentage_list = np.array(channel_watch_percentage_list)
    channel_relative_engagement_list = np.array(channel_relative_engagement_list)

    print('short videos mean watch percentage: {0:.4f}'.format(np.mean(channel_watch_percentage_list[channel_duration_list <= 1000])))
    print('short videos std watch percentage: {0:.4f}'.format(np.std(channel_watch_percentage_list[channel_duration_list <= 1000])))
    print('long videos mean watch percentage: {0:.4f}'.format(np.mean(channel_watch_percentage_list[channel_duration_list > 1000])))
    print('long videos mean watch percentage: {0:.4f}'.format(np.std(channel_watch_percentage_list[channel_duration_list > 1000])))
    print('channel mean watch percentage: {0:.4f}'.format(np.mean(channel_watch_percentage_list)))

    print('short videos mean relative engagement: {0:.4f}'.format(np.mean(channel_relative_engagement_list[channel_duration_list <= 1000])))
    print('short videos std relative engagement: {0:.4f}'.format(np.std(channel_relative_engagement_list[channel_duration_list <= 1000])))
    print('long videos mean relative engagement: {0:.4f}'.format(np.mean(channel_relative_engagement_list[channel_duration_list > 1000])))
    print('long videos mean relative engagement: {0:.4f}'.format(np.std(channel_relative_engagement_list[channel_duration_list > 1000])))
    print('channel mean relative engagement: {0:.4f}'.format(np.mean(channel_relative_engagement_list)))

    # == == == == == == == == Part 3: Plot engagement metrics within the same channel == == == == == == == == #
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.scatter(channel_duration_list, channel_watch_percentage_list, lw=1.5, s=50, facecolors='none', edgecolors='b', marker='o')
    ax2.scatter(channel_duration_list, channel_relative_engagement_list, lw=1.5, s=50, color='r', marker='x')

    ax1.set_ylabel('watch percentage $\\bar \mu_{30}$', fontsize=17)
    ax2.set_ylabel('relative engagement $\\bar \eta_{30}$', fontsize=17)

    for ax in [ax1, ax2]:
        ax.set_ylim([0, 1.05])
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(FuncFormatter(exponent))
        ax.set_xlabel('video duration (sec) $D$', fontsize=17)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    plt.savefig('../images/fig5_within_channel.pdf', bbox_inches='tight')
    plt.show()
