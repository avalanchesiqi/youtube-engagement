#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script to plot Figure 8, show plausible effect of predicting via relative engagement.
UCy-taFzDCV-XcpTBa3pF68w: PBABowling

Usage: python plot_fig8_relative_engagement_effect.py
Time: ~1M
"""

import sys, os, pickle, platform
import numpy as np

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer
from utils.converter import to_watch_percentage


def build_reputation_features(arr):
    arr = np.array(arr)
    return [len(arr)/51, np.mean(arr), np.std(arr),
            np.min(arr), np.percentile(arr, 25), np.median(arr), np.percentile(arr, 75), np.max(arr)]


def predict(duration, reputation_features, params):
    all_features = [1, np.log10(duration)]
    all_features.extend(reputation_features)
    return sum([a*b for a, b in zip(all_features, params)])


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to plot plausible effect of predicting via relative engagement...')
    timer = Timer()
    timer.start()

    channel_id = 'UCy-taFzDCV-XcpTBa3pF68w'
    channel_title = 'PBABowling'

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    engagement_map_loc = '../data/engagement_map.p'
    if not os.path.exists(engagement_map_loc):
        print('Engagement map not generated, start with generating engagement map first in ../data dir!')
        print('Exit program...')
        sys.exit(1)

    engagement_map = pickle.load(open(engagement_map_loc, 'rb'))
    split_keys = np.array(engagement_map['duration'])

    channel_view_train_path = '../engagement_prediction/train_data_channel_view/UC{0}.p'.format(channel_id[2])
    if not os.path.exists(channel_view_train_path):
        print('>>> No train data in channel view found! Run construct_channel_view_dataset.py first!')
        sys.exit(1)

    channel_train_videos = pickle.load(open(channel_view_train_path, 'rb'))[channel_id]

    channel_duration_list = []
    channel_watch_percentage_list = []
    channel_relative_engagement_list = []

    for video in channel_train_videos:
        _, _, duration, tails = video.split('\t', 3)
        _, wp30, re30, _, _, _ = tails.rsplit('\t', 5)
        channel_duration_list.append(int(duration))
        channel_watch_percentage_list.append(float(wp30))
        channel_relative_engagement_list.append(float(re30))

    # == == == == == == == == Part 3: Set model parameters == == == == == == == == #
    # model parameters are returned from shared reputation predictors in format of intercept_ and coef_
    # for watch percentage
    wp_model_params = [0.5074039791274337, -0.15008378, -0.00051926, 0.25462437, -0.12992794,
                       -0.00045746, 0.13600957, 0.10778634, 0.13124552, 0.07857673]
    wp_reputation_features = build_reputation_features(channel_watch_percentage_list)

    # for relative engagement
    re_model_params = [0.04803607742076138, -0.0187069675, -0.000229044755, 0.465002888, -0.0403513233,
                       0.0319145686, 0.173823259, 0.101924752, 0.168012995, 0.0568290748]

    re_reputation_features = build_reputation_features(channel_relative_engagement_list)

    # == == == == == == == == Part 4: Plot function == == == == == == == == #
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111)

    ax1.scatter(channel_duration_list, channel_watch_percentage_list, lw=1, s=50, facecolors='none', edgecolors='k', marker='o')

    x_min, x_max = ax1.get_xlim()
    x_min = max(x_min, 10)
    x_axis = np.arange(x_min, x_max, 3)

    y_axis_wp_direct = [predict(x, wp_reputation_features, wp_model_params) for x in x_axis]
    ax1.plot(x_axis, y_axis_wp_direct, 'b--', label=r'Predict $\bar \mu_{30}$ directly')

    y_axis_re_direct = [predict(x, re_reputation_features, re_model_params) for x in x_axis]
    y_axis_wp_converted = []
    for x, y in zip(x_axis, y_axis_re_direct):
        y_axis_wp_converted.append(to_watch_percentage(engagement_map, x, y, lookup_keys=split_keys))
    ax1.plot(x_axis, y_axis_wp_converted, 'r', label=r'Predict $\bar \eta_{30}$ then map to $\bar \mu_{30}$')

    ax1.set_ylabel('average watch percentage $\\bar \mu_{30}$', fontsize=14)

    ax1.set_ylim([0, 1.05])
    ax1.set_xscale('log')
    ax1.set_xlabel('video duration (sec) $D$', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.legend(loc='lower left', fontsize=14, frameon=False)

    timer.stop()

    plt.tight_layout()
    plt.savefig('../images/fig8_relative_engagement_effect.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()
