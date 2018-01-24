#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Scripts to plot Figure 4, compare relative engagement within the same channel."""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def exponent(x, pos):
    'The two args are the value and tick position'
    return '%1.0f' % (x)


x_formatter = FuncFormatter(exponent)

if __name__ == '__main__':
    channel_title = 'PBABowling'
    video_durations = [5829, 5714, 110, 4729, 147, 133, 69, 5558, 59, 77, 4601, 4995, 11, 146, 118, 5723, 122, 58, 3590,
                       5845, 145, 132, 9, 160, 5052]
    video_wps = [0.223242118095, 0.162085090236, 0.783407427569, 0.192458562525, 0.718550898839, 0.810546604288,
                 0.887627214594, 0.213057611597, 0.896746577457, 0.859189434917, 0.244711173999, 0.25872527202, 1.0,
                 0.835765561308, 0.759799808495, 0.226426821865, 0.76698949307, 0.824802053483, 0.223251758158,
                 0.202020913515, 0.6755161209, 0.756837066897, 1.0, 0.788271236785, 0.196453396022]
    video_res = [0.833, 0.664, 0.908, 0.712, 0.866, 0.965, 0.958, 0.809, 0.939, 0.944, 0.826, 0.87, 1.0, 0.984, 0.886,
                 0.827, 0.907, 0.787, 0.712, 0.789, 0.783, 0.902, 1.0, 0.966, 0.714]

    video_durations = np.array(video_durations)
    video_wps = np.array(video_wps)
    video_res = np.array(video_res)

    print('short videos mean watch percentage', np.mean(video_wps[video_durations < 1000]),
          'short videos std watch percentage', np.std(video_wps[video_durations < 1000]))
    print('long videos mean watch percentage', np.mean(video_wps[video_durations > 1000]),
          'long videos mean watch percentage', np.std(video_wps[video_durations > 1000]))
    print('channel mean watch percentage', np.mean(video_wps))

    print('short videos mean relative engagement', np.mean(video_res[video_durations < 1000]),
          'short videos std relative engagement', np.std(video_res[video_durations < 1000]))
    print('long videos mean relative engagement', np.mean(video_res[video_durations > 1000]),
          'long videos mean relative engagement', np.std(video_res[video_durations > 1000]))
    print('channel mean relative engagement', np.mean(video_res))

    # plot function
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.scatter(video_durations, video_wps, lw=1.5, s=50, facecolors='none', edgecolors='b', marker='o')
    ax2.scatter(video_durations, video_res, lw=1.5, s=50, color='r', marker='x')

    ax1.set_ylabel('watch percentage $\\bar \mu_{30}$', fontsize=17)
    ax2.set_ylabel('relative engagement $\\bar \eta_{30}$', fontsize=17)

    for ax in [ax1, ax2]:
        ax.set_ylim([0, 1.05])
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(x_formatter)
        ax.set_xlabel('video duration (sec) $D$', fontsize=17)
        ax.tick_params(axis='both', which='major', labelsize=14)
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
    # ax1.set_title('(c)', fontsize=20)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    plt.savefig('../images/fig4_within_channel.pdf', bbox_inches='tight')
    plt.show()
