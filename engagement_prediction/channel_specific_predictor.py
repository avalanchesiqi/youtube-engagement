#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Predict engagement metrics from channel specific predictor.

Target: predict watch percentage
Usage: python channel_specific_predictor.py -i ./ -o ./output -f wp
Time: ~20M

Target: predict relative engagement
Usage: python channel_specific_predictor.py -i ./ -o ./output -f re
Time: ~25M
"""

import os, sys, argparse, pickle
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, write_dict_to_pickle
from utils.ridge_regressor import RidgeRegressor


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    timer = Timer()
    timer.start()

    predict_result_dict = {}
    # for one channel, minimal videos in training dataset so the predictor can be built
    k = 5

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
        print('>>> Start to predict watch percentage with all features in channel specific predictor...')
    elif target == 're':
        is_re = True
        print('>>> Start to predict relative engagement with all features in channel specific predictor...')
    else:
        print('>>> Error: Unknown prediction target! It mush be wp or re!')
        print('>>> Exit...')
        print(sys.exit(1))

    train_loc = os.path.join(input_dir, 'train_data_channel_view')
    test_loc = os.path.join(input_dir, 'test_data_channel_view')
    if not (os.path.exists(train_loc) and os.path.exists(test_loc)):
        print('>>> Error: Did not find train or test dataset in channel view!')
        print('>>> Exit...')
        print(sys.exit(1))

    os.makedirs(output_dir, exist_ok=True)

    # == == == == == == == == Part 3: Start training == == == == == == == == #
    for subdir, _, files in os.walk(test_loc):
        for f in files:
            print('>>> Start to predict test channel {0}...'.format(os.path.join(subdir, f)))
            test_channel_cluster = pickle.load(open(os.path.join(subdir, f), 'rb'))
            train_channel_cluster = pickle.load(open(os.path.join(train_loc, f), 'rb'))
            # if we have observed this channel before, minimal observations: k
            test_channel_list = test_channel_cluster.keys()
            train_channel_list = train_channel_cluster.keys()
            for channel_id in test_channel_list:
                if channel_id in train_channel_list and len(train_channel_cluster[channel_id]) >= k:
                    train_videos = train_channel_cluster[channel_id]
                    test_videos = test_channel_cluster[channel_id]

                    # get topic encoding
                    topic_dict = {}
                    topic_cnt = 0
                    for topics in [x.split('\t')[7] for x in train_videos]:
                        if not (topics == '' or topics == 'NA'):
                            for topic in topics.split(','):
                                if topic not in topic_dict:
                                    topic_dict[topic] = topic_cnt
                                    topic_cnt += 1

                    # get channel history
                    train_matrix = []
                    for train_video in train_videos:
                        row = np.zeros(1+1+category_cnt+lang_cnt+topic_cnt+1)
                        vid, publish, duration, definition, category, detect_lang, _, topics, _, _, wp30, re30, _ = train_video.rstrip().split('\t', 12)
                        row[0] = np.log10(int(duration))
                        if definition == '1':
                            row[1] = 1
                        if category in category_dict:
                            row[2 + category_dict[category]] = 1
                        if detect_lang in lang_dict:
                            row[2 + category_cnt + lang_dict[detect_lang]] = 1
                        if not (topics == '' or topics == 'NA'):
                            topics = topics.split(',')
                            for topic in topics:
                                row[2 + category_cnt + lang_cnt + topic_dict[topic]] = 1
                        target = [wp30, re30][is_re]
                        row[-1] = float(target)
                        train_matrix.append(row)
                    train_matrix = np.array(train_matrix)

                    test_matrix = []
                    test_vids = []
                    for test_video in test_videos:
                        row = np.zeros(1+1+category_cnt+lang_cnt+topic_cnt+1)
                        vid, publish, duration, definition, category, detect_lang, _, topics, _, _, wp30, re30, _ = test_video.rstrip().split('\t', 12)
                        row[0] = np.log10(int(duration))
                        if definition == '1':
                            row[1] = 1
                        if category in category_dict:
                            row[2 + category_dict[category]] = 1
                        if detect_lang in lang_dict:
                            row[2 + category_cnt + lang_dict[detect_lang]] = 1
                        if not (topics == '' or topics == 'NA'):
                            topics = topics.split(',')
                            for topic in topics:
                                if topic in topic_dict:
                                    row[2 + category_cnt + lang_cnt + topic_dict[topic]] = 1
                        target = [wp30, re30][is_re]
                        row[-1] = float(target)
                        test_matrix.append(row)
                        test_vids.append(vid)
                    test_matrix = np.array(test_matrix)

                    # predict test data from customized ridge regressor
                    test_yhat = RidgeRegressor(train_matrix, test_matrix, verbose=False).predict()

                    predict_result_dict.update({vid: pred for vid, pred in zip(test_vids, test_yhat)})

    timer.stop()

    # write to pickle file
    to_write = True
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        write_dict_to_pickle(dict=predict_result_dict,
                             path=os.path.join(output_dir, '{0}_csp_predictor_{1}.p'.format(['wp', 're'][is_re], k)))
