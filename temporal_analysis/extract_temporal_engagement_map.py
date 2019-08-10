#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to construct temporal engagement map from temporal engagement data.
# Fields: vid, category, duration, watch_percentage, watch_time

Usage: python extract_temporal_engagement_map.py -i ./temporal_engagement_data -o ./temporal_engagement_map
Time: ~1H
"""

import sys, os, pickle, argparse
import numpy as np
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer


def plot_map_from_file(input_path, output_dir):
    age = os.path.basename(input_path)[-6: -4]
    duration_engagement_tuple = []
    duration_cnt_dict = defaultdict(int)

    with open(input_path, 'r') as fin:
        for line in fin:
            vid, category, duration, watch_percentage, watch_time = line.rstrip().split(',')
            duration = int(duration)
            watch_percentage = float(watch_percentage)
            watch_time = float(watch_time)
            duration_engagement_tuple.append((duration, watch_percentage, np.log10(watch_time)))
            duration_cnt_dict[duration] += 1

    # == == == == == == == == Part 1: Build wp matrix based on duration splits == == == == == == == == #
    # sort by duration in ascent order
    sorted_duration_engagement_tuple = sorted(duration_engagement_tuple, key=lambda x: x[0])

    # get ideal split point in log scale
    even_split_points = [10**x for x in np.linspace(1, 5, bin_number)]

    # put videos in correct bins
    wp_bin_matrix = []
    wp_individual_bin = []
    wt_bin_matrix = []
    wt_individual_bin = []
    bin_idx = 0
    duration_splits = []
    # put dur-wp tuple in the correct bin
    for duration, watch_percentage, watch_time in sorted_duration_engagement_tuple:
        if duration > even_split_points[bin_idx]:
            bin_idx += 1
            # must contains at least 50 videos
            if len(wp_individual_bin) >= 50:
                wp_bin_matrix.append(np.array(wp_individual_bin))
                wp_individual_bin = []
                wt_bin_matrix.append(np.array(wt_individual_bin))
                wt_individual_bin = []
                duration_splits.append(even_split_points[bin_idx])
        wp_individual_bin.append(watch_percentage)
        wt_individual_bin.append(watch_time)
    if len(wp_individual_bin) > 0:
        wp_bin_matrix.append(np.array(wp_individual_bin))
        wt_bin_matrix.append(np.array(wt_individual_bin))

    # == == == == == == == == Part 2: Store engagement map offline == == == == == == == == #
    engagement_map = {'duration': duration_splits}
    for i in range(len(wp_bin_matrix)):
        engagement_map[i] = [np.percentile(wp_bin_matrix[i], j / 10) for j in range(1000)]
    pickle.dump(engagement_map, open(os.path.join(output_dir, 'engagement_map_day{0}.p'.format(age)), 'wb'))


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to extract temporal engagement maps...')
    timer = Timer()
    timer.start()

    bin_number = 1000

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file dir of temporal engagement data', required=True)
    parser.add_argument('-o', '--output', help='output file dir of temporal engagement maps', required=True)
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

    for subdir, _, files in os.walk(input_dir):
        for f in files:
            print('>>> Start to extract engagement map: {0}...'.format(os.path.join(subdir, f)))
            plot_map_from_file(os.path.join(subdir, f), output_dir)
    print('>>> Extracted all data!')

    timer.stop()
