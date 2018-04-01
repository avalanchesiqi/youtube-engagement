#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Predict watch percentage from channel reputation.

Usage: python reputation_predictor.py -i ../ -o ./output
Time: ~30M
"""

import os, sys, time, datetime, argparse
from collections import defaultdict
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.helper import write_dict_to_pickle
from utils.ridge_regressor import RidgeRegressor


def _load_data(filepath):
    """ Load features space for channel reputation predictor.
    """
    matrix = []
    vids = []
    print('>>> Start to load file {0}...'.format(filepath))
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            row = np.zeros(10)
            vid, _, duration, _, _, _, channel_id, _, _, _, wp30, _, _ = line.rstrip().split('\t', 12)
            if channel_id in channel_wp_dict:
                row[0] = np.log10(int(duration))
                # training dataset spans over 51 days
                row[1] = len(channel_wp_dict[channel_id]) / 51
                row[2] = np.mean(channel_wp_dict[channel_id])
                row[3] = np.std(channel_wp_dict[channel_id])
                row[4] = np.min(channel_wp_dict[channel_id])
                row[5] = np.percentile(channel_wp_dict[channel_id], 25)
                row[6] = np.median(channel_wp_dict[channel_id])
                row[7] = np.percentile(channel_wp_dict[channel_id], 75)
                row[8] = np.max(channel_wp_dict[channel_id])
                row[9] = float(wp30)
                matrix.append(row)
                vids.append(vid)
    return matrix, vids


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to predict watch percentage with channel reputation...')
    start_time = time.time()

    train_channel_watch_percentage_path = './output/train_channel_watch_percentage.txt'
    if not os.path.exists(train_channel_watch_percentage_path):
        print('>>> No channel watch percentage found! Run extract_channel_reputation.py first!')
        sys.exit(1)

    channel_wp_dict = defaultdict(list)
    with open(train_channel_watch_percentage_path, 'r') as fin:
        for line in fin:
            channel_id, wp30 = line.rstrip().split('\t')
            channel_wp_dict[channel_id].append(float(wp30))

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file dir of formatted dataset', required=True)
    parser.add_argument('-o', '--output', help='output file dir of predictor result', required=True)
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    train_loc = os.path.join(input_dir, 'train_data')
    test_loc = os.path.join(input_dir, 'test_data')

    print('>>> Start to load training dataset...')
    train_matrix = []
    for subdir, _, files in os.walk(train_loc):
        for f in files:
            train_matrix.extend(_load_data(os.path.join(subdir, f))[0])
    train_matrix = np.array(train_matrix)

    print('>>> Start to load test dataset...')
    test_matrix = []
    test_vids = []
    for subdir, _, files in os.walk(test_loc):
        for f in files:
            matrix, vids = _load_data(os.path.join(subdir, f))
            test_matrix.extend(matrix)
            test_vids.extend(vids)
    test_matrix = np.array(test_matrix)

    print('>>> Finish loading all data!')

    # predict test data from customized ridge regressor
    test_yhat = RidgeRegressor(train_matrix, test_matrix).predict()

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    # write to pickle file
    to_write = True
    predict_result_dict = {vid: pred for vid, pred in zip(test_vids, test_yhat)}
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        write_dict_to_pickle(dict=predict_result_dict, path=os.path.join(output_dir, 'reputation_predictor.p'))
