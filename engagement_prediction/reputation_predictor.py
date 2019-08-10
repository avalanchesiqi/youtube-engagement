#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Predict engagement metrics from channel reputation.

Target: predict watch percentage
Usage: python reputation_predictor.py -i ./ -o ./output -f wp
Time: ~50M

Target: predict relative engagement
Usage: python reputation_predictor.py -i ./ -o ./output -f re
Time: ~50M
"""

import os, sys, argparse
from collections import defaultdict
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, write_dict_to_pickle
from utils.ridge_regressor import RidgeRegressor


def _load_data(filepath, is_re):
    """ Load features space for channel reputation predictor.
    """
    matrix = []
    vids = []
    print('>>> Start to load file {0}...'.format(filepath))
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            row = np.zeros(10)
            vid, _, duration, _, _, _, channel_id, _, _, _, wp30, re30, _ = line.rstrip().split('\t', 12)
            if channel_id in channel_engagement_dict:
                row[0] = np.log10(int(duration))
                # training dataset spans over 51 days
                row[1] = len(channel_engagement_dict[channel_id]) / 51
                row[2] = np.mean(channel_engagement_dict[channel_id])
                row[3] = np.std(channel_engagement_dict[channel_id])
                row[4] = np.min(channel_engagement_dict[channel_id])
                row[5] = np.percentile(channel_engagement_dict[channel_id], 25)
                row[6] = np.median(channel_engagement_dict[channel_id])
                row[7] = np.percentile(channel_engagement_dict[channel_id], 75)
                row[8] = np.max(channel_engagement_dict[channel_id])
                target = [wp30, re30][is_re]
                row[-1] = float(target)
                matrix.append(row)
                vids.append(vid)
    return matrix, vids


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    timer = Timer()
    timer.start()

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file dir of formatted dataset', required=True)
    parser.add_argument('-o', '--output', help='output file dir of predictor result', required=True)
    parser.add_argument('-f', '--function', help='choose prediction target', required=True)
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    target = args.function

    if target == 'wp':
        is_re = False
        print('>>> Start to predict watch percentage with channel reputation...')
    elif target == 're':
        is_re = True
        print('>>> Start to predict relative engagement with channel reputation...')
    else:
        print('>>> Error: Unknown prediction target! It mush be wp or re!')
        print('>>> Exit...')
        print(sys.exit(1))

    train_loc = os.path.join(input_dir, 'train_data')
    test_loc = os.path.join(input_dir, 'test_data')
    if not (os.path.exists(train_loc) and os.path.exists(test_loc)):
        print('>>> Error: Did not find train or test dataset!')
        print('>>> Exit...')
        print(sys.exit(1))

    os.makedirs(output_dir, exist_ok=True)

    train_channel_reputation_path = './output/train_channel_{0}.txt'.format(['watch_percentage', 'relative_engagement'][is_re])
    if not os.path.exists(train_channel_reputation_path):
        print('>>> No channel past reputation found! Run extract_channel_reputation.py first!')
        sys.exit(1)

    channel_engagement_dict = defaultdict(list)
    with open(train_channel_reputation_path, 'r') as fin:
        for line in fin:
            channel_id, engagement_metric = line.rstrip().split('\t')
            channel_engagement_dict[channel_id].append(float(engagement_metric))

    # == == == == == == == == Part 3: Start training == == == == == == == == #
    print('>>> Start to load training dataset...')
    train_matrix = []
    for subdir, _, files in os.walk(train_loc):
        for f in files:
            train_matrix.extend(_load_data(os.path.join(subdir, f), is_re)[0])
    train_matrix = np.array(train_matrix)

    print('>>> Start to load test dataset...')
    test_matrix = []
    test_vids = []
    for subdir, _, files in os.walk(test_loc):
        for f in files:
            matrix, vids = _load_data(os.path.join(subdir, f), is_re)
            test_matrix.extend(matrix)
            test_vids.extend(vids)
    test_matrix = np.array(test_matrix)

    print('>>> Finish loading all data!')

    # predict test data from customized ridge regressor
    test_yhat = RidgeRegressor(train_matrix, test_matrix).predict(show_params=True)

    timer.stop()

    # write to pickle file
    to_write = True
    predict_result_dict = {vid: pred for vid, pred in zip(test_vids, test_yhat)}
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        write_dict_to_pickle(dict=predict_result_dict,
                             path=os.path.join(output_dir, '{0}_reputation_predictor.p'.format(['wp', 're'][is_re])))
