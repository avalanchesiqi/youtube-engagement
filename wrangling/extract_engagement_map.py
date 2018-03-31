#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script to extract engagement maps from given formatted dataset and store offline.
Here the age cutoff is day 30. For temporal engagement map, refer to extract_temporal_engagement_map.py

Usage: python extract_engagement_map.py -i ../data/formatted_tweeted_videos -o ../data/engagement_map.p
Time: ~8M
"""

import sys, os, time, datetime, pickle, argparse
import numpy as np
from collections import defaultdict


def get_engagement_stats_from_file(filepath):
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            _, _, duration, _, _, _, _, _, view, _, wp30, _ = line.split('\t', 11)
            duration = int(duration)
            wp30 = float(wp30)
            duration_engagement_tuple.append((duration, wp30, np.log10(duration * wp30)))
            duration_cnt_dict[duration] += 1


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to extract engagement map...')
    start_time = time.time()

    bin_number = 1000
    duration_engagement_tuple = []
    duration_cnt_dict = defaultdict(int)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file dir of formatted dataset', required=True)
    parser.add_argument('-o', '--output', help='output file path of engagement map', required=True)
    args = parser.parse_args()

    input_dir = args.input
    output_path = args.output

    if not os.path.exists(input_dir):
        print('>>> Input file dir does not exist!')
        print('>>> Exit...')
        sys.exit(1)

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    for subdir, _, files in os.walk(input_dir):
        for f in files:
            print('>>> Start to load data: {0}...'.format(os.path.join(subdir, f)))
            get_engagement_stats_from_file(os.path.join(subdir, f))
    print('>>> Finish loading all data!')

    # == == == == == == == == Part 3: Build wp matrix based on duration splits == == == == == == == == #
    # sort by duration in ascent order
    sorted_duration_engagement_tuple = sorted(duration_engagement_tuple, key=lambda x: x[0])

    # get duration split point
    duration_list = sorted(duration_cnt_dict.keys())
    even_split_points = list(np.linspace(1, 5, bin_number))

    # put videos in correct bins
    wp_bin_matrix = []
    wp_individual_bin = []
    wt_bin_matrix = []
    wt_individual_bin = []
    bin_idx = 0
    duration_splits = []
    # put dur-wp tuple in the correct bin
    for duration, wp30, wt30 in sorted_duration_engagement_tuple:
        if np.log10(duration) > even_split_points[bin_idx]:
            bin_idx += 1
            # must contains at least 50 videos
            if len(wp_individual_bin) >= 50:
                wp_bin_matrix.append(wp_individual_bin)
                wp_individual_bin = []
                wt_bin_matrix.append(wt_individual_bin)
                wt_individual_bin = []
                duration_splits.append(duration - 1)
        wp_individual_bin.append(wp30)
        wt_individual_bin.append(wt30)
    if len(wp_individual_bin) > 0:
        wp_bin_matrix.append(np.array(wp_individual_bin))
        wt_bin_matrix.append(np.array(wt_individual_bin))

    # == == == == == == == == Part 4: Store engagement map offline == == == == == == == == #
    engagement_map = {'duration': duration_splits}
    for i in range(len(duration_splits) + 1):
        engagement_map[i] = [np.percentile(wp_bin_matrix[i], j / 10) for j in range(1000)]
    pickle.dump(engagement_map, open('../data/engagement_map.p', 'wb'))

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])
