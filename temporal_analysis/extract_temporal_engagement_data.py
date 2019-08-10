#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to extract data to build temporal engagement maps.

Usage: python extract_temporal_engagement_data.py -i ../data/formatted_tweeted_videos -o ./temporal_engagement_data
Time: ~1H
"""

import os, sys, argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, read_as_int_array, read_as_float_array


def extract_engagement_data_from_file(filepath, age, handles, threshold=100):
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            vid, _, duration, _, category, _, _, _, _, _, _, days, daily_view, daily_watch = line.rstrip().split('\t')
            duration = int(duration)
            days = read_as_int_array(days, delimiter=',', truncated=age)
            daily_view = read_as_int_array(daily_view, delimiter=',', truncated=age)
            daily_watch = read_as_float_array(daily_watch, delimiter=',', truncated=age)
            for i in range(age):
                views = np.sum(daily_view[days <= i])
                watches = np.sum(daily_watch[days <= i])
                if views >= threshold:
                    watch_time = watches * 60 / views
                    if watch_time > duration:
                        watch_time = duration
                    watch_percentage = watch_time / duration
                    handles[i].write('{0},{1},{2},{3},{4}\n'.format(vid, category, duration, watch_percentage, watch_time))


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to extract temporal engagement data...')
    timer = Timer()
    timer.start()

    age_range = 30
    view_threshold = 100

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file dir of formatted dataset', required=True)
    parser.add_argument('-o', '--output', help='output file dir of temporal engagement data', required=True)
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    if not os.path.exists(input_dir):
        print('>>> Input file dir does not exist!')
        print('>>> Exit...')
        sys.exit(1)

    if os.path.exists(output_dir):
        print('>>> Output file dir already exists, rename, check or backup it before starting new job!')
        print('>>> Exit...')
        sys.exit(1)
    else:
        os.mkdir(output_dir)

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    # open a series of file handle
    output_handles = [open('{0}/engagement_data_day{1:>02}.csv'.format(output_dir, t), 'w') for t in range(1, age_range+1)]

    for subdir, _, files in os.walk(input_dir):
        for f in files:
            print('>>> Start to extract data: {0}...'.format(os.path.join(subdir, f)))
            extract_engagement_data_from_file(os.path.join(subdir, f), age_range, output_handles, view_threshold)
    print('>>> Finish loading all data!')

    # close all file handles
    for handle in output_handles:
        handle.close()

    timer.stop()
