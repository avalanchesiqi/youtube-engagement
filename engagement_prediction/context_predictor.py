#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Predict engagement metrics from video context.

Target: predict watch percentage
Usage: python context_predictor.py -i ./ -o ./output -f wp
Time: ~5M

Target: predict relative engagement
Usage: python context_predictor.py -i ./ -o ./output -f re
Time: ~5M
"""

import os, sys, argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, write_dict_to_pickle
from utils.ridge_regressor import RidgeRegressor


def _load_data(filepath, is_re):
    """ Load features space for context predictor.
    """
    matrix = []
    vids = []
    print('>>> Start to load file {0}...'.format(filepath))
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            row = np.zeros(1 + 1 + category_cnt + lang_cnt + 1)
            vid, publish, duration, definition, category, detect_lang, _, _, _, _, wp30, re30, _ = line.rstrip().split('\t', 12)
            vids.append(vid)
            row[0] = np.log10(int(duration))
            if definition == '1':
                row[1] = 1
            if category in category_dict:
                row[2 + category_dict[category]] = 1
            if detect_lang in lang_dict:
                row[2 + category_cnt + lang_dict[detect_lang]] = 1
            target = [wp30, re30][is_re]
            row[-1] = float(target)
            matrix.append(row)
    return matrix, vids


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    timer = Timer()
    timer.start()

    category_dict = {'1': 0, '2': 1, '10': 2, '15': 3, '17': 4, '19': 5, '20': 6, '22': 7, '23': 8, '24': 9,
                     '25': 10, '26': 11, '27': 12, '28': 13, '29': 14, '30': 15, '43': 16}
    category_cnt = len(category_dict)
    lang_dict = {'af': 0, 'ar': 1, 'bg': 2, 'bn': 3, 'ca': 4, 'cs': 5, 'cy': 6, 'da': 7, 'de': 8, 'el': 9, 'en': 10,
                 'es': 11, 'et': 12, 'fa': 13, 'fi': 14, 'fr': 15, 'gu': 16, 'he': 17, 'hi': 18, 'hr': 19, 'hu': 20,
                 'id': 21, 'it': 22, 'ja': 23, 'kn': 24, 'ko': 25, 'lt': 26, 'lv': 27, 'mk': 28, 'ml': 29, 'mr': 30,
                 'ne': 31, 'nl': 32, 'no': 33, 'pa': 34, 'pl': 35, 'pt': 36, 'ro': 37, 'ru': 38, 'sk': 39, 'sl': 40,
                 'so': 41, 'sq': 42, 'sv': 43, 'sw': 44, 'ta': 45, 'te': 46, 'th': 47, 'tl': 48, 'tr': 49, 'uk': 50,
                 'ur': 51, 'vi': 52, 'zh-cn': 53, 'zh-tw': 54}
    lang_cnt = len(lang_dict)

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
        print('>>> Start to predict watch percentage with video context...')
    elif target == 're':
        is_re = True
        print('>>> Start to predict relative engagement with video context...')
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
    test_yhat = RidgeRegressor(train_matrix, test_matrix).predict()

    timer.stop()

    # write to pickle file
    to_write = True
    predict_result_dict = {vid: pred for vid, pred in zip(test_vids, test_yhat)}
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        write_dict_to_pickle(dict=predict_result_dict,
                             path=os.path.join(output_dir, '{0}_context_predictor.p'.format(['wp', 're'][is_re])))
