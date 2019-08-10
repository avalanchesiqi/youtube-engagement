#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert watch percentage from relative engagement dataframe.

Example rows
           Vid     True    Duration     Context       Topic      CTopic  Reputation          All         CSP  VDuration
0  KcgjkCDPOco    0.498         0.5    0.608499    0.610826    0.667708    0.871146     0.878522    0.878522       1212
1  oydbUUFZNPQ    0.301         0.5    0.405562    0.635107    0.515533    0.424186     0.489899    0.435074        350
2  RUAKJSxfgW0    0.945         0.5    0.462427    0.737947    0.738719    0.699488     0.715835    0.450224        254
3  U45p1d_zQEs    0.512         0.5    0.504788    0.491619    0.501147    0.127934     0.142407    0.000000        209
4  wjdjztvb9Hc    0.988         0.5    0.523769    0.635107    0.515533    0.994331     0.489899    0.489899        160
"""

import os, sys, pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer
from utils.converter import to_watch_percentage


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to convert predicted relative engagement results from watch percentage results...')
    timer = Timer()
    timer.start()

    engagement_map_loc = '../data/engagement_map.p'
    if not os.path.exists(engagement_map_loc):
        print('Engagement map not generated, start with generating engagement map first in ../../data dir!')
        print('Exit program...')
        sys.exit(1)

    engagement_map = pickle.load(open(engagement_map_loc, 'rb'))
    split_keys = np.array(engagement_map['duration'])

    # load pandas dataframe if exists
    re_dataframe_path = './output/predicted_re_df.csv'
    if os.path.exists(re_dataframe_path):
        re_data_f = pd.read_csv(re_dataframe_path, sep='\t')
    else:
        print('Relative engagement dataframe not found!')
        sys.exit(1)

    wp_dataframe_path = './output/predicted_wp_df.csv'
    if os.path.exists(wp_dataframe_path):
        wp_data_f = pd.read_csv(wp_dataframe_path, sep='\t')
    else:
        print('Watch percentage dataframe not found!')
        sys.exit(1)

    # mae and r2 list
    mae_list = []
    r2_list = []
    name_list = ['Duration', 'Context', 'Topic', 'CTopic', 'Reputation', 'All', 'CSP']
    for name in name_list:
        mae_list.append(mean_absolute_error(wp_data_f['True'], wp_data_f[name]))
        r2_list.append(r2_score(wp_data_f['True'], wp_data_f[name]))

        converted_wp = to_watch_percentage(engagement_map, re_data_f['VDuration'].tolist(), re_data_f[name].tolist(), lookup_keys=split_keys)
        mae_list.append(mean_absolute_error(wp_data_f['True'], converted_wp))
        r2_list.append(r2_score(wp_data_f['True'], converted_wp))

    for i in range(len(name_list)):
        print('\n>>> {2} MAE scores for wp and converted wp: {0} - {1}'.format(mae_list[2*i], mae_list[2*i+1], name_list[i]))
        print('>>> {2} R2  scores for wp and converted wp: {0} - {1}'.format(r2_list[2*i], r2_list[2*i+1], name_list[i]))

    timer.stop()
