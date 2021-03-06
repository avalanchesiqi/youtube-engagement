#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script to plot Figure 1, scatter plot for videos from three YouTube channels.
UC0aCRHZm9xOCvFoxWAmgS2w: Blunt Force Truth
UCp0hYYBW6IMayGgR-WeoCvQ: TheEllenShow
UCHn2-DNS5t4tEXqBK5bHmTQ: KEEMI

Usage: python plot_fig1_teaser.py
Time: ~2M
"""

import os, sys, platform
from collections import defaultdict

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to extract information about selected channels...')
    timer = Timer()
    timer.start()

    channel_ids = ['UC0aCRHZm9xOCvFoxWAmgS2w', 'UCp0hYYBW6IMayGgR-WeoCvQ', 'UCHn2-DNS5t4tEXqBK5bHmTQ']
    labels = ['Blunt Force Truth', 'TheEllenShow', 'KEEMI']

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    input_loc = '../data/formatted_tweeted_videos'
    channel_view_dict = defaultdict(list)
    channel_wp_dict = defaultdict(list)

    for subdir, _, files in os.walk(input_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                fin.readline()
                for line in fin:
                    _, _, duration, _, _, _, channel, _, view30, _, wp30, _ = line.rstrip().split('\t', 11)
                    if channel in channel_ids:
                        channel_view_dict[channel].append(int(view30))
                        channel_wp_dict[channel].append(float(wp30))

    # == == == == == == == == Part 3: Plot videos of selected channels == == == == == == == == #
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(1, 1, 1)

    styles = ['o', 'x', '^']
    colors = ['b', 'r', 'g']

    for idx, channel in enumerate(channel_ids):
        if styles[idx] == 'x':
            ax1.scatter(channel_view_dict[channel], channel_wp_dict[channel], lw=1, s=50,
                        color=colors[idx], label=labels[idx], marker=styles[idx])
        else:
            ax1.scatter(channel_view_dict[channel], channel_wp_dict[channel], lw=1, s=50,
                        facecolors='none', edgecolors=colors[idx], label=labels[idx], marker=styles[idx])

    ax1.set_xlabel('total views in the first 30 days', fontsize=16)
    ax1.set_ylabel('average watch percentage', fontsize=16)
    ax1.set_ylim([0, 1])
    ax1.set_xscale('log')
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.legend(loc='lower left', fontsize=16, scatterpoints=2, frameon=False)

    timer.stop()

    plt.tight_layout()
    plt.savefig('../images/fig1_teaser.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()
