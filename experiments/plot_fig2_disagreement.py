#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to plot Figure 2, spearman rank correlation for top 1000 videos in terms of total views and watch time.

Input: ../../production_data/new_tweeted_dataset_norm/
Usage: python plot_fig2_disagreement.py
Time: ~2M
"""

from __future__ import division, print_function
import os
from scipy import stats
import operator
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')


def plot_spearman(ax, view_rank_dict, watch_rank_dict, color, linestyle, label):
    sorted_view_rank = sorted(view_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:1000]
    sorted_watch_rank = sorted(watch_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:1000]

    x_axis = []
    y_axis = []
    # iterate from 50 to 1000, with gap 10
    for i in range(50, 1001, 10):
        view_set = set([item[0] for item in sorted_view_rank[:i]])
        watch_set = set([item[0] for item in sorted_watch_rank[:i]])
        union_list = list(view_set.union(watch_set))
        view_rank = [view_rank_dict[x] for x in union_list]
        watch_rank = [watch_rank_dict[x] for x in union_list]

        x_axis.append(i)
        y_axis.append(stats.spearmanr(view_rank, watch_rank)[0])

    ax.plot(x_axis, y_axis, color=color, linestyle=linestyle, label=label, lw=2)


def plot_scatter(ax, view_rank_dict, watch_rank_dict, color='k'):
    sorted_view_rank = sorted(view_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:100]
    sorted_watch_rank = sorted(watch_rank_dict.items(), key=operator.itemgetter(1), reverse=True)[:100]

    view_set = set([item[0] for item in sorted_view_rank])
    watch_set = set([item[0] for item in sorted_watch_rank])
    union_list = list(view_set.union(watch_set))
    vid_view_watch_tripet = [(vid, view_rank_dict[vid], watch_rank_dict[vid]) for vid in union_list]

    sorted_by_view = [x[0] for x in sorted(vid_view_watch_tripet, key=lambda x: x[1], reverse=True)]
    sorted_by_watch = [x[0] for x in sorted(vid_view_watch_tripet, key=lambda x: x[2], reverse=True)]

    view_rank = []
    watch_rank = []
    for vid in union_list:
        view_rank.append(sorted_by_view.index(vid) + 1)
        watch_rank.append(sorted_by_watch.index(vid) + 1)

    print(stats.spearmanr(view_rank, watch_rank))

    ax.scatter(view_rank, watch_rank, s=1, c=color)


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    total_view_rank_dict = {}
    total_watch_rank_dict = {}
    music_view_rank_dict = {}
    music_watch_rank_dict = {}
    news_view_rank_dict = {}
    news_watch_rank_dict = {}

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    input_doc = '../../production_data/new_tweeted_dataset_norm/'
    for subdir, _, files in os.walk(input_doc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                fin.readline()
                for line in fin:
                    vid, dump = line.rstrip().split('\t', 1)
                    view30 = float(dump.split('\t')[7])
                    watch30 = float(dump.split('\t')[8])

                    total_view_rank_dict[vid] = view30
                    total_watch_rank_dict[vid] = watch30

                    if f.startswith('10'):
                        music_view_rank_dict[vid] = view30
                        music_watch_rank_dict[vid] = watch30

                    if f.startswith('25'):
                        news_view_rank_dict[vid] = view30
                        news_watch_rank_dict[vid] = watch30
            print('>>> Loading data: {0} done!'.format(os.path.join(subdir, f)))

    # == == == == == == == == Part 3: Plot figures == == == == == == == == #
    plt.figure(figsize=(9, 6))
    ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    ax3 = plt.subplot2grid((2, 3), (1, 2))

    # ax1
    plot_spearman(ax1, total_view_rank_dict, total_watch_rank_dict, color='k', linestyle='-', label=r'\textsc{Tweeted videos}')
    plot_spearman(ax1, music_view_rank_dict, music_watch_rank_dict, color='r', linestyle='--', label='Music')
    plot_spearman(ax1, news_view_rank_dict, news_watch_rank_dict, color='b', linestyle='--', label='News')
    ax1.plot((100, 100), (ax1.get_xlim()[0], ax1.get_xlim()[1]), 'k:')
    ax1.set_ylim([-1, 1])
    ax1.set_xlabel('top $n$ videos', fontsize=14)
    ax1.set_ylabel("Spearman's $\\rho$", fontsize=14)
    ax1.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.legend(loc='lower right', handlelength=1, frameon=False, fontsize=16)
    ax1.set_title('(a)', fontsize=16)

    # ax2, music, get the rank value
    plot_scatter(ax2, music_view_rank_dict, music_watch_rank_dict, 'r')
    ax2.set_xlabel('total view rank', fontsize=14)
    ax2.set_ylabel('total watch rank', fontsize=14)
    ax2.set_title('(b) Music', fontsize=16)

    # ax3, news, get the rank value
    plot_scatter(ax3, news_view_rank_dict, news_watch_rank_dict, 'b')
    ax3.set_xlabel('total view rank', fontsize=14)
    ax3.set_ylabel('total watch rank', fontsize=14)
    ax3.set_title('(c) News', fontsize=16)

    # for ax in [ax1, ax2, ax3]:
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust()
    plt.savefig('../images/fig2_disagreement.pdf', bbox_inches='tight')
    plt.show()
