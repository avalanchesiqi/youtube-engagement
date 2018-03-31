#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Scripts to plot Figure 7, show a plausible example of relative engagement predictor."""

from __future__ import print_function, division
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from utils.converter import to_watch_percentage


def exponent(x, pos):
    'The two args are the value and tick position'
    return '%1.0f' % (x)


x_formatter = FuncFormatter(exponent)


def build_features(arr):
    arr = np.array(arr)
    return [np.mean(arr), np.std(arr), np.min(arr), np.percentile(arr, 25), np.median(arr), np.percentile(arr, 75),
            np.max(arr)]


def predict(duration, features, params):
    all_features = [1, np.log10(duration), 0]
    all_features.extend(features)
    return sum([a*b for a,b in zip(all_features, params)])


if __name__ == '__main__':
    engagement_map = pickle.load(open('../data/engagement_map.p', 'rb'))

    channel_title = 'PBABowling'
    train_durations = [110, 4729, 147, 133, 69, 5558, 59, 77, 4601, 4995, 11, 146, 118, 5723, 122, 58, 3590,
                       5845, 145, 132, 9, 160, 5052]
    train_wps = [0.783407427569, 0.192458562525, 0.718550898839, 0.810546604288,
                 0.887627214594, 0.213057611597, 0.896746577457, 0.859189434917, 0.244711173999, 0.25872527202, 1.0,
                 0.835765561308, 0.759799808495, 0.226426821865, 0.76698949307, 0.824802053483, 0.223251758158,
                 0.202020913515, 0.6755161209, 0.756837066897, 1.0, 0.788271236785, 0.196453396022]
    train_res = [0.908, 0.712, 0.866, 0.965, 0.958, 0.809, 0.939, 0.944, 0.826, 0.87, 1.0, 0.984, 0.886,
                 0.827, 0.907, 0.787, 0.712, 0.789, 0.783, 0.902, 1.0, 0.966, 0.714]
    test_durations = [5829, 5714]
    test_wps = [0.223242118095, 0.162085090236]
    test_res = [0.833, 0.664]

    # wp model
    wp_params = [0.507314933758, -0.14996086, -0.00079967, 0.52440454, -0.2005125,
                 -0.0404418, 0.03659694, 0.05410489, 0.07009169, 0.06347822]
    print(wp_params)
    wp_features = build_features(train_wps)

    # re model
    re_params = [0.0480400443602, -1.87085307e-02, -2.33563629e-04, 4.65018563e-01, -4.03505247e-02,
                 3.19127276e-02, 1.73814285e-01, 1.01926135e-01, 1.68007435e-01, 5.68276519e-02]
    print(re_params)
    re_features = build_features(train_res)

    # plot function
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111)

    ax1.scatter(train_durations, train_wps, lw=1, s=50, facecolors='none', edgecolors='k', marker='o')
    # ax1.scatter(test_durations, test_wps, lw=1, s=50, facecolors='g', edgecolors='g', marker='^')

    x_min, x_max = ax1.get_xlim()
    x_min = max(x_min, 10)
    x_axis = np.arange(x_min, x_max, 3)
    y_axis1 = [predict(x, wp_features, wp_params) for x in x_axis]
    ax1.plot(x_axis, y_axis1, 'b--', label=r'Predict $\bar \mu_{30}$')

    y_axis2_tmp = [predict(x, re_features, re_params) for x in x_axis]
    y_axis2 = []
    for x, y in zip(x_axis, y_axis2_tmp):
        y_axis2.append(to_watch_percentage(engagement_map, x, y))
    print(y_axis2)
    ax1.plot(x_axis, y_axis2, 'r', label=r'Predict $\bar \eta_{30}$ then map to $\bar \mu_{30}$')

    # y_axis3 = []
    # for x in x_axis:
    #     y_axis3.append(to_watch_percentage(engagement_map, x, 0.5))
    # print(y_axis3)
    # ax1.plot(x_axis, y_axis3, 'g:', label=r'Predict $\bar \mu_{30}$ via $\bar \eta_{30}=0.5$')

    ax1.set_ylabel('average watch percentage $\\bar \mu_{30}$', fontsize=14)

    ax1.set_ylim([0, 1.05])
    ax1.set_xscale('log')
    ax1.xaxis.set_major_formatter(x_formatter)
    ax1.set_xlabel('video duration (sec) $D$', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    # ax1.spines['right'].set_visible(False)
    # ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.legend(loc='lower left', fontsize=14, frameon=False)

    plt.tight_layout()
    plt.savefig('../images/fig7_predictor_example.pdf', bbox_inches='tight')
    plt.savefig('../../yt-engagement/image/fig7_predictor_example.pdf', bbox_inches='tight')
    plt.show()
