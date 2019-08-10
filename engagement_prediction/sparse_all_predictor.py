#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Predict engagement metrics from all features in sparse matrix.

Target: predict watch percentage
Usage: python sparse_all_predictor.py -i ./ -o ./output -f wp
Time: ~3H30M

Target: predict relative engagement
Usage: python sparse_all_predictor.py -i ./ -o ./output -f re
Time: ~3H30M
"""

import os, sys, argparse
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, strify, write_dict_to_pickle
from utils.ridge_regressor import RidgeRegressor


def _load_data(filepath, is_re):
    """ Load all features space.
    """
    matrix = []
    print('>>> Start to load file {0}...'.format(filepath))
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            vid, _, duration, definition, category, detect_lang, channel_id, topics, _, _, wp30, re30, _ = line.rstrip().split('\t', 12)
            if channel_id in channel_engagement_dict:
                reputation_vector = strify([len(channel_engagement_dict[channel_id])/51,
                                            np.mean(channel_engagement_dict[channel_id]),
                                            np.std(channel_engagement_dict[channel_id]),
                                            np.min(channel_engagement_dict[channel_id]),
                                            np.percentile(channel_engagement_dict[channel_id], 25),
                                            np.median(channel_engagement_dict[channel_id]),
                                            np.percentile(channel_engagement_dict[channel_id], 75),
                                            np.max(channel_engagement_dict[channel_id])])
                target = [wp30, re30][is_re]
                row = [vid, duration, definition, category, detect_lang, topics, reputation_vector, target]
                matrix.append(row)
    return matrix


def _build_sparse_matrix(row_idx, duration, definition, category, detect_lang, topics, topic_dict, reputation_vector, keys=None):
    row_list = []
    col_list = []
    value_list = []

    row_list.append(row_idx)
    col_list.append(0)
    value_list.append(np.log10(int(duration)))

    if definition == '1':
        row_list.append(row_idx)
        col_list.append(1)
        value_list.append(1)

    if not category == '44':
        row_list.append(row_idx)
        col_list.append(2 + category_dict[category])
        value_list.append(1)

    if not detect_lang == 'NA':
        row_list.append(row_idx)
        col_list.append(2 + category_cnt + lang_dict[detect_lang])
        value_list.append(1)

    topics = topics.split(',')
    for topic in topics:
        if keys is None or (keys is not None and topic in keys):
            row_list.append(row_idx)
            col_list.append(2 + category_cnt + lang_cnt + topic_dict[topic])
            value_list.append(1)

    n_topic = len(topic_dict)
    reputation_vector = map(float, reputation_vector.split(','))
    for i, reputation_metric in enumerate(reputation_vector):
        row_list.append(row_idx)
        col_list.append(2 + category_cnt + lang_cnt + n_topic + i)
        value_list.append(float(reputation_metric))
    return row_list, col_list, value_list


def vectorize_train_data(data):
    train_row = []
    train_col = []
    train_value = []
    train_y = []
    topic_dict = {}
    topic_cnt = 0
    row_idx = 0

    for _, _, _, _, _, topics, _, _ in data:
        if not (topics == '' or topics == 'NA'):
            topics = topics.split(',')
            for topic in topics:
                if topic not in topic_dict:
                    topic_dict[topic] = topic_cnt
                    topic_cnt += 1

    for _, duration, definition, category, detect_lang, topics, reputation_vector, target in data:
        if not (topics == '' or topics == 'NA'):
            row_list, col_list, value_list = _build_sparse_matrix(row_idx, duration, definition, category, detect_lang, topics, topic_dict, reputation_vector)
            train_row.extend(row_list)
            train_col.extend(col_list)
            train_value.extend(value_list)
            train_y.append(float(target))
            row_idx += 1
    return coo_matrix((train_value, (train_row, train_col)), shape=(row_idx, 2 + category_cnt + lang_cnt + topic_cnt + 8)), train_y, topic_dict


def vectorize_test_data(data, topic_dict):
    test_vids = []
    test_row = []
    test_col = []
    test_value = []
    test_y = []
    n_topic = len(topic_dict)
    topic_keys = list(topic_dict.keys())
    row_idx = 0

    for vid, duration, definition, category, detect_lang, topics, reputation_vector, target in data:
        if not (topics == '' or topics == 'NA'):
            row_list, col_list, value_list = _build_sparse_matrix(row_idx, duration, definition, category, detect_lang, topics, topic_dict, reputation_vector, keys=topic_keys)
            test_row.extend(row_list)
            test_col.extend(col_list)
            test_value.extend(value_list)
            test_y.append(float(target))
            row_idx += 1
            test_vids.append(vid)
    return coo_matrix((test_value, (test_row, test_col)), shape=(row_idx, 2 + category_cnt + lang_cnt + n_topic + 8)), test_y, test_vids


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to predict engagement metrics with all features in sparse matrix...')
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
        print('>>> Start to predict watch percentage with all features in sparse matrix...')
    elif target == 're':
        is_re = True
        print('>>> Start to predict relative engagement with all features in sparse matrix...')
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
            train_matrix.extend(_load_data(os.path.join(subdir, f), is_re))

    print('>>> Start to load test dataset...')
    test_matrix = []
    for subdir, _, files in os.walk(test_loc):
        for f in files:
            test_matrix.extend(_load_data(os.path.join(subdir, f), is_re))

    print('>>> Finish loading all data!')

    # predict test data from customized ridge regressor
    test_yhat, test_vids = RidgeRegressor(train_matrix, test_matrix).predict_from_sparse(vectorize_train_data,
                                                                                         vectorize_test_data)

    timer.stop()

    # write to pickle file
    to_write = True
    predict_result_dict = {vid: pred for vid, pred in zip(test_vids, test_yhat)}
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        write_dict_to_pickle(dict=predict_result_dict,
                             path=os.path.join(output_dir, '{0}_sparse_all_predictor.p'.format(['wp', 're'][is_re])))
