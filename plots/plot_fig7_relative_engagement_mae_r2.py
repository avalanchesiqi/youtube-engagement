#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script to plot Figure 7, prediction results of relative engagement.

Usage: python plot_fig7_relative_engagement_mae_r2.py
Time: ~1M
"""

import sys, os, time, datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def plot_barchart(mae_list, r2_list):
    # generate barchart plot
    fig = plt.figure(figsize=(8, 3))
    width = 0.4
    n = len(mae_list)
    ind = np.arange(n)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    cornflower_blue = '#6495ed'
    tomato = '#ff6347'

    ax1.bar(ind, mae_list, width, edgecolor=['k']*n, color=cornflower_blue, lw=1.5)
    ax2.bar(ind, r2_list, width, edgecolor=['k']*n, color=tomato, lw=1.5)

    ax1.set_ylim([0, 0.32])
    ax2.set_ylim([0, 1])
    ax1.set_yticks([0, 0.1, 0.2])
    ax2.set_yticks([0.2, 0.6, 1.0])
    ax1.set_ylabel(r'$MAE$', fontsize=16)
    ax2.set_ylabel(r'$R^2$', fontsize=16)

    for label in ax1.get_xticklabels()[:]:
        label.set_visible(False)
    ax2.set_xticklabels(('', 'D', 'C', 'T', 'C+T', 'R', 'All', 'CSP'))

    for tick, label in zip(ind, ax1.get_xticklabels()):
        ax1.text(ind[tick], mae_list[tick] + 0.01, [str(np.round(x, 4)) for x in mae_list][tick],
                 horizontalalignment='center', color='k', fontsize=14)
    for tick, label in zip(ind, ax2.get_xticklabels()):
        ax2.text(ind[tick], r2_list[tick] + 0.01, [str(np.round(x, 4)) for x in r2_list][tick],
                 horizontalalignment='center', color='k', fontsize=14)

    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    ax1.set_title(r'(a) Predict relative engagement $\bar \eta_{30}$', fontsize=18)

    # ax1.legend([plt.Rectangle((0, 0), 1, 1, fc=cornflower_blue, ec='k'),
    #             plt.Rectangle((0, 0), 1, 1, fc=tomato, ec='k')],
    #            ['MAE', r'$R^2$'], fontsize=14, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -2.3),
    #            fancybox=True, shadow=True, ncol=2)


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to plot prediction results of relative engagement...')
    start_time = time.time()

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    # load pandas dataframe if exists
    dataframe_path = '../engagement_prediction/output/predicted_re_df.csv'
    if os.path.exists(dataframe_path):
        data_f = pd.read_csv(dataframe_path, sep='\t')
    else:
        print('Data frame not generated yet, go back to construct_pandas_frame.py!')
        sys.exit(1)

    m, n = data_f.shape
    print('>>> Final dataframe size {0} instances with {1} features'.format(m, n))
    print('>>> Header of final dataframe')
    print(data_f.head())

    # mae and r2 list
    mae_list = []
    r2_list = []
    for name in ['Duration', 'Context', 'Topic', 'CTopic', 'Reputation', 'All', 'CSP']:
        mae_list.append(mean_absolute_error(data_f['True'], data_f[name]))
        r2_list.append(r2_score(data_f['True'], data_f[name]))
    print('\n>>> MAE scores: ', mae_list)
    print('>>> R2 scores: ', r2_list)

    r2_list[0] = 0.000

    plot_barchart(mae_list, r2_list)

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    plt.tight_layout(rect=[0.01, 0, 1, 0.99])
    plt.savefig('../images/fig7_predict_relative_engagement.pdf', bbox_inches='tight')
    plt.show()
