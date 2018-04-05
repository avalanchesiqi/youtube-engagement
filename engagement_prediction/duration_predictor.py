#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Predict engagement metrics from video duration.

Target: predict watch percentage
Usage: python duration_predictor.py -i ./ -o ./output -f wp
Time: ~1M

Target: predict relative engagement
Usage: python duration_predictor.py -i ./ -o ./output -f re
Time: ~1M
"""

import os, sys, time, datetime, pickle, argparse
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import write_dict_to_pickle
from utils.converter import to_watch_percentage


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    start_time = time.time()

    test_vids = []
    test_duration = []
    true_engagement = []
    guess_engagement = []

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
        print('>>> Start to predict watch percentage with video duration...')
    elif target == 're':
        is_re = True
        print('>>> Start to predict relative engagement with video duration...')
    else:
        print('>>> Error: Unknown prediction target! It mush be wp or re!')
        print('>>> Exit...')
        print(sys.exit(1))

    test_loc = os.path.join(input_dir, 'test_data')
    if not os.path.exists(test_loc):
        print('>>> Error: Did not find test dataset!')
        print('>>> Exit...')
        print(sys.exit(1))

    os.makedirs(output_dir, exist_ok=True)

    engagement_map_loc = '../data/engagement_map.p'
    if not os.path.exists(engagement_map_loc):
        print('Engagement map not generated, start with generating engagement map first in ../data dir!')
        print('Exit program...')
        sys.exit(1)

    engagement_map = pickle.load(open(engagement_map_loc, 'rb'))
    split_keys = np.array(engagement_map['duration'])

    # == == == == == == == == Part 3: Start training == == == == == == == == #
    for subdir, _, files in os.walk(test_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                fin.readline()
                for line in fin:
                    vid, _, duration, _, _, _, _, _, _, _, wp30, re30, _ = line.rstrip().split('\t', 12)
                    test_vids.append(vid)
                    duration = int(duration)
                    test_duration.append(duration)
                    target = float([wp30, re30][is_re])
                    true_engagement.append(target)
                    random_guess = 0.5
                    if is_re:
                        guess_engagement.append(random_guess)
                    else:
                        guess_engagement.append(to_watch_percentage(engagement_map, duration, random_guess, lookup_keys=split_keys))

    print('>>> Predict {0} on duration...'.format(['watch percentage', 'relative engagement'][is_re]))
    print('>>> MAE on test set: {0:.4f}'.format(mean_absolute_error(true_engagement, guess_engagement)))
    print('>>> R2 on test set: {0:.4f}'.format(r2_score(true_engagement, guess_engagement)))
    print('=' * 79)

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    # write to pickle file
    to_write = True
    true_result_dict = {vid: true for vid, true in zip(test_vids, true_engagement)}
    predict_result_dict = {vid: pred for vid, pred in zip(test_vids, guess_engagement)}
    test_duration_dict = {vid: duration for vid, duration in zip(test_vids, test_duration)}
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(test_vids)))
        write_dict_to_pickle(dict=true_result_dict,
                             path=os.path.join(output_dir, '{0}_true_predictor.p'.format(['wp', 're'][is_re])))
        write_dict_to_pickle(dict=predict_result_dict,
                             path=os.path.join(output_dir, '{0}_duration_predictor.p'.format(['wp', 're'][is_re])))
        if not os.path.exists(os.path.join(output_dir, 'test_duration.p')):
            write_dict_to_pickle(dict=test_duration_dict, path=os.path.join(output_dir, 'test_duration.p'))

