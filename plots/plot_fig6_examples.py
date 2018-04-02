#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to plot relative engagement temporal fitting examples.
video 1: XIB8Z_hASOs
video 2: hxUh6dS5Q_Q

Usage: python plot_fig6_examples.py
Time: ~1M
"""

import os, sys, time, datetime
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import read_as_float_array, read_as_int_array


def func_powerlaw(x, a, b, c):
    return a * (x ** b) + c


def fit_with_powerlaw(x, y):
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    try:
        params, _ = curve_fit(func_powerlaw, x, y, maxfev=10000)
        return params
    except:
        pass


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to plot temporal relative engagement fitting for exemplary videos...')
    start_time = time.time()
    age = 30
    ts = np.arange(1, age + 1)

    fig = plt.figure(figsize=(16, 5))
    fig_cnt = 0
    with open('../temporal_analysis/sliding_engagement_dynamics.csv', 'r') as fin:
        for line in fin:
            if fig_cnt == 2:
                break

            vid, days, _, re_list = line.rstrip().split('\t')

            if vid == 'XIB8Z_hASOs':
                fig_idx = 0
                daily_view = [981, 208, 570, 653, 786, 650, 783, 730, 716, 842, 812, 990, 915, 1355, 1538, 749, 751,
                              800, 849, 969, 1015, 888, 696, 863, 897, 1039, 1348, 1659, 1259, 1559]
                fig_cnt += 1
            elif vid == 'hxUh6dS5Q_Q':
                fig_idx = 1
                daily_view = [20634, 15162, 6925, 5132, 4348, 3625, 3437, 5255, 6226, 6021, 10104, 7183, 11172, 10655,
                              15246, 15990, 17911, 14262, 12128, 11120, 7191, 5882, 3867, 5271, 2352, 2004, 2677, 2817,
                              3266, 2968]
                fig_cnt += 1
            else:
                continue

            days = read_as_int_array(days, delimiter=',')
            re_list = read_as_float_array(re_list, delimiter=',')

            # power-law fitting
            a, b, c = fit_with_powerlaw(days, re_list)

            print(vid, r'model: {0:.3f}t^{1:.3f}+{2:.3f}'.format(a, b, c))

            # == == == == == == == == Part 2: Plot fitting result == == == == == == == == #
            to_plot = True
            if to_plot:
                ax1 = fig.add_subplot(1, 2, 1 + fig_idx)
                ax2 = ax1.twinx()
                ax1.plot(ts, daily_view, 'b-')
                ax2.plot(ts, re_list, 'k--')
                ax2.plot(ts, [func_powerlaw(x, a, b, c) for x in ts], 'r-')

                if fig_idx == 0:
                    ax1.set_ylabel(r'daily view $x_v$', color='b', fontsize=18)
                if fig_idx == 1:
                    ax2.set_ylabel(r'smoothed relative engagement $\bar \eta_{t-6:t}$', color='k', fontsize=16)

                ax1.set_xlabel('video age (day)', fontsize=18)
                ax1.set_xlim([0, 31])
                ax1.set_ylim(ymin=max(0, ax1.get_ylim()[0]))
                ax2.set_ylim([0, 1])
                ax1.tick_params('y', colors='b')
                ax2.tick_params('y', colors='k')

                annotated_str = r'ID: {0}'.format(vid)
                annotated_str += '\nmodel: '
                if c > 0:
                    annotated_str += r'${%.2f}t^{%.2f}+{%.2f}$' % (a, b, c)
                else:
                    annotated_str += r'${%.2f}t^{%.2f}-{%.2f}$' % (a, b, -c)

                if fig_idx == 0:
                    ax2.text(30.5, 0.05, annotated_str, horizontalalignment='right', fontsize=20)
                else:
                    ax2.text(30.5, 0.82, annotated_str, horizontalalignment='right', fontsize=20)

                ax2.set_xticks([0, 10, 20, 30])
                display_min = int(np.floor(min(daily_view) / 100) * 100)
                display_max = int(np.ceil(max(daily_view) / 100) * 100)
                ax1.set_yticks([display_min, (display_min+display_max)/2, display_max])
                ax2.set_yticks([0.0, 0.5, 1.0])
                for ax in [ax1, ax2]:
                    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90)
                    ax.tick_params(axis='both', which='major', labelsize=16)

    plt.title('(c)', fontsize=24)
    plt.legend([plt.Line2D((0, 1), (0, 0), color='b'),
                plt.Line2D((0, 1), (0, 0), color='k', linestyle='--'),
                plt.Line2D((0, 1), (0, 0), color='r')],
               ['Observed view series', 'Observed relative engagement', 'Fitted relative engagement'],
               fontsize=20, frameon=False, handlelength=1,
               loc='lower center', bbox_to_anchor=(0, -0.35), ncol=3)

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    plt.tight_layout(rect=[0, 0.06, 1, 1], h_pad=0)
    plt.savefig('../images/fig6_fit_examples.pdf', bbox_inches='tight')
    plt.show()
