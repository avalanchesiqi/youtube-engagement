#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script to plot Figure 2, Spearman rank correlation for top 1000 videos in terms of total views and watch time.

Usage: python plot_fig2_disagreement.py
Time: ~2M
"""

import os, sys, operator, platform
from scipy import stats

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer


def plot_spearman(ax, view_rank_dict, watch_rank_dict, color, linestyle, label):
    cap = 1000
    sorted_view_rank = sorted(view_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:cap]
    sorted_watch_rank = sorted(watch_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:cap]

    x_axis = []
    y_axis = []
    # iterate from 50 to cap+1, with gap 10
    for i in range(50, cap+1, 10):
        view_set = set([item[0] for item in sorted_view_rank[:i]])
        watch_set = set([item[0] for item in sorted_watch_rank[:i]])
        union_list = list(view_set.union(watch_set))
        view_rank = [view_rank_dict[x] for x in union_list]
        watch_rank = [watch_rank_dict[x] for x in union_list]

        x_axis.append(i)
        y_axis.append(stats.spearmanr(view_rank, watch_rank)[0])

    ax.plot(x_axis, y_axis, color=color, linestyle=linestyle, label=label, lw=2)


def plot_scatter(ax, view_rank_dict, watch_rank_dict, color='k', cap=100):
    sorted_view_rank = sorted(view_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:cap]
    sorted_watch_rank = sorted(watch_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:cap]

    view_set = set([item[0] for item in sorted_view_rank])
    watch_set = set([item[0] for item in sorted_watch_rank])
    union_list = list(view_set.union(watch_set))
    vid_view_watch_triplet = [(vid, view_rank_dict[vid], watch_rank_dict[vid]) for vid in union_list]

    sorted_by_view = [x[0] for x in sorted(vid_view_watch_triplet, key=lambda x: x[1], reverse=True)]
    sorted_by_watch = [x[0] for x in sorted(vid_view_watch_triplet, key=lambda x: x[2], reverse=True)]

    view_rank = []
    watch_rank = []
    for vid in union_list:
        view_rank.append(sorted_by_view.index(vid) + 1)
        watch_rank.append(sorted_by_watch.index(vid) + 1)

    print('>>> Spearman correlation:', stats.spearmanr(view_rank, watch_rank))
    ax.scatter(view_rank, watch_rank, s=1, c=color)


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to plot Spearman correlation between top views and top watched videos...')
    timer = Timer()
    timer.start()

    num_display_top = 100
    total_view_rank_dict = {}
    total_watch_rank_dict = {}
    music_view_rank_dict = {}
    music_watch_rank_dict = {}
    news_view_rank_dict = {}
    news_watch_rank_dict = {}

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    input_loc = '../data/formatted_tweeted_videos'
    for subdir, _, files in os.walk(input_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                fin.readline()
                for line in fin:
                    vid, _, _, _, _, _, _, _, view30, watch30, _ = line.rstrip().split('\t', 10)
                    view30 = float(view30)
                    watch30 = float(watch30)

                    total_view_rank_dict[vid] = view30
                    total_watch_rank_dict[vid] = watch30
                    if f.startswith('music'):
                        music_view_rank_dict[vid] = view30
                        music_watch_rank_dict[vid] = watch30
                    if f.startswith('news'):
                        news_view_rank_dict[vid] = view30
                        news_watch_rank_dict[vid] = watch30

    # == == == == == == == == Part 3: Plot figures == == == == == == == == #
    plt.figure(figsize=(9, 6))
    ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    ax3 = plt.subplot2grid((2, 3), (1, 2))

    # ax1, spearman correlation with various top value
    plot_spearman(ax1, total_view_rank_dict, total_watch_rank_dict, color='k', linestyle='-', label='TWEETED VIDEOS')
    plot_spearman(ax1, music_view_rank_dict, music_watch_rank_dict, color='r', linestyle='--', label='Music')
    plot_spearman(ax1, news_view_rank_dict, news_watch_rank_dict, color='b', linestyle='--', label='News')
    ax1.plot([num_display_top, num_display_top], [-1, 1], 'k:')
    ax1.set_ylim([-1, 1])
    ax1.set_xlabel('top $n$ videos', fontsize=14)
    ax1.set_ylabel("Spearman's $\\rho$", fontsize=14)
    ax1.set_xticks([100, 300, 500, 700, 900])
    ax1.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.legend(loc='lower right', handlelength=1, frameon=False, fontsize=16)
    ax1.set_title('(a)', fontsize=16)

    # ax2, music, get the rank value
    plot_scatter(ax2, music_view_rank_dict, music_watch_rank_dict, 'r', cap=num_display_top)
    ax2.set_xticks([0, 50, 100])
    ax2.set_yticks([0, 50, 100])
    ax2.set_xlabel('total view rank', fontsize=14)
    ax2.set_ylabel('total watch rank', fontsize=14)
    ax2.set_title(r'(b) Music at $n$=100', fontsize=16)

    # ax3, news, get the rank value
    plot_scatter(ax3, news_view_rank_dict, news_watch_rank_dict, 'b', cap=num_display_top)
    ax3.set_xticks([0, 50, 100, 150])
    ax3.set_yticks([0, 50, 100, 150])
    ax3.set_xlabel('total view rank', fontsize=14)
    ax3.set_ylabel('total watch rank', fontsize=14)
    ax3.set_title(r'(c) News at $n$=100', fontsize=16)

    timer.stop()

    plt.tight_layout()
    plt.subplots_adjust()
    plt.savefig('../images/fig2_disagreement.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()
