#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Predict relative engagement from video duration.

Usage: python duration_predictor.py -i ../ -o ./output
Time: ~1M
"""

import os, sys, time, datetime, argparse
from sklearn.metrics import mean_absolute_error, r2_score

from utils.helper import write_dict_to_pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to predict relative engagement with video duration...')
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file dir of formatted dataset', required=True)
    parser.add_argument('-o', '--output', help='output file dir of predictor result', required=True)
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    test_vids = []
    test_duration = []
    true_re = []
    guess_re = []

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    test_loc = os.path.join(input_dir, 'test_data')

    for subdir, _, files in os.walk(test_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                fin.readline()
                for line in fin:
                    vid, _, duration, dump = line.rstrip().split('\t', 3)
                    test_vids.append(vid)
                    duration = int(duration)
                    test_duration.append(duration)
                    re30 = float(dump.split('\t')[8])
                    true_re.append(re30)
                    random_guess = 0.5
                    guess_re.append(random_guess)

    print('>>> Predict relative engagement on duration...')
    print('>>> MAE on test set: {0:.4f}'.format(mean_absolute_error(true_re, guess_re)))
    print('>>> R2 on test set: {0:.4f}'.format(r2_score(true_re, guess_re)))
    print('=' * 79)

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    # write to pickle file
    to_write = True
    true_result_dict = {vid: true for vid, true in zip(test_vids, true_re)}
    predict_result_dict = {vid: pred for vid, pred in zip(test_vids, guess_re)}
    test_duration_dict = {vid: duration for vid, duration in zip(test_vids, test_duration)}
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(test_vids)))
        write_dict_to_pickle(dict=true_result_dict, path=os.path.join(output_dir, 'true_predictor.p'))
        write_dict_to_pickle(dict=predict_result_dict, path=os.path.join(output_dir, 'duration_predictor.p'))
        write_dict_to_pickle(dict=test_duration_dict, path=os.path.join(output_dir, 'test_duration.p'))
