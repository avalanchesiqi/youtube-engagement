#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to extract cumulative and sliding engagement dynamics in TWEETED VIDEOS dataset from day 1 to day 30.
In sliding window engagement dynamics, we apply a minimal view threshold of 100.

Usage: python extract_temporal_engagement_dynamics.py -i ../data/formatted_tweeted_videos
Time: ~15H
"""


import os, sys, pickle, argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.converter import to_relative_engagement
from utils.helper import Timer, read_as_int_array, read_as_float_array, strify


def extract_engagement_dynamics_from_file(filepath, engagement_map_series, split_key_series, window_size, min_view=100):
    age = len(engagement_map_series)
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            vid, _, duration, _, _, _, _, _, _, _, _, days, daily_view, daily_watch = line.rstrip().split('\t')
            duration = int(duration)
            days = read_as_int_array(days, delimiter=',', truncated=age)
            daily_view = read_as_int_array(daily_view, delimiter=',', truncated=age)
            daily_watch = read_as_float_array(daily_watch, delimiter=',', truncated=age)
            cum_day_list = []
            cum_wp_list = []
            cum_re_list = []
            sliding_day_list = []
            sliding_wp_list = []
            sliding_re_list = []
            for i in range(age):
                # cumulative watch percentage and relative engagement
                cum_views = np.sum(daily_view[days <= i])
                cum_watches = np.sum(daily_watch[days <= i])
                if cum_views >= min_view:
                    cum_wp = cum_watches * 60 / cum_views / duration
                    if cum_wp > 1:
                        cum_wp = 1
                    cum_re = to_relative_engagement(engagement_map_series[i], duration, cum_wp, lookup_keys=split_key_series[i])
                    cum_day_list.append(i)
                    cum_wp_list.append(cum_wp)
                    cum_re_list.append(cum_re)

                # sliding window watch percentage and relative engagement
                if i < window_size:
                    sliding_views = np.sum(daily_view[days <= i])
                    sliding_watches = np.sum(daily_watch[days <= i])
                else:
                    sliding_views = np.sum(daily_view[(i - window_size < days) & (days <= i)])
                    sliding_watches = np.sum(daily_watch[(i - window_size < days) & (days <= i)])
                if sliding_views >= min_view:
                    sliding_wp = sliding_watches * 60 / sliding_views / duration
                    if sliding_wp > 1:
                        sliding_wp = 1
                    sliding_re = to_relative_engagement(engagement_map_series[i], duration, sliding_wp, lookup_keys=split_key_series[i])
                    sliding_day_list.append(i)
                    sliding_wp_list.append(sliding_wp)
                    sliding_re_list.append(sliding_re)

            # write to output files
            if len(cum_day_list) > 0:
                cum_output.write('{0}\t{1}\t{2}\t{3}\n'.format(vid, strify(cum_day_list, delimiter=','),
                                                               strify(cum_wp_list, delimiter=','),
                                                               strify(cum_re_list, delimiter=',')))
            if len(sliding_day_list) > 0:
                sliding_output.write('{0}\t{1}\t{2}\t{3}\n'.format(vid, strify(sliding_day_list, delimiter=','),
                                                                   strify(sliding_wp_list, delimiter=','),
                                                                   strify(sliding_re_list, delimiter=',')))


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to extract temporal engagement dynamics...')
    timer = Timer()
    timer.start()

    age_range = 30
    window_size = 7
    min_view = 100
    engagement_map_series = []
    split_key_series = []

    cmu_engagement_dynamics = {}
    sliding_engagement_dynamics = {}

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file dir of formatted dataset', required=True)
    args = parser.parse_args()

    input_dir = args.input
    cum_output = open('./cum_engagement_dynamics.csv', 'w')
    sliding_output = open('./sliding_engagement_dynamics.csv', 'w')

    if not os.path.exists(input_dir):
        print('>>> Input file dir does not exist!')
        print('>>> Exit...')
        sys.exit(1)

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    for i in range(1, age_range+1):
        map_path = './temporal_engagement_map/engagement_map_day{0:>02}.p'.format(i)
        engagement_map = pickle.load(open(map_path, 'rb'))
        split_keys = engagement_map['duration']
        engagement_map_series.append(engagement_map)
        split_key_series.append(split_keys)

    # == == == == == == == == Part 3: Extract cumulative and sliding windows engagement == == == == == == == == #
    for subdir, _, files in os.walk(input_dir):
        for f in files:
            print('>>> Start to extract data: {0}...'.format(os.path.join(subdir, f)))
            extract_engagement_dynamics_from_file(os.path.join(subdir, f), engagement_map_series, split_key_series, window_size, min_view)
    print('>>> Finish extracting all data!')

    cum_output.close()
    sliding_output.close()

    timer.stop()
