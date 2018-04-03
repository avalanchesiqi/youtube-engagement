#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Predict engagement metrics from freebase topics in sparse matrix.

Target: predict watch percentage
Usage: python sparse_topic_predictor.py -i ./ -o ./output -f wp
Time: ~2H

Target: predict relative engagement
Usage: python sparse_topic_predictor.py -i ./ -o ./output -f re
Time: ~2H
"""

import os, sys, time, datetime, argparse
import numpy as np
from scipy.sparse import coo_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import write_dict_to_pickle
from utils.ridge_regressor import RidgeRegressor


def _load_data(filepath, is_re):
    """ Load features space for topic predictor.
    """
    matrix = []
    print('>>> Start to load file {0}...'.format(filepath))
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            vid, _, duration, _, _, _, _, topics, _, _, wp30, re30, _ = line.rstrip().split('\t', 12)
            target = [wp30, re30][is_re]
            row = [vid, duration, topics, target]
            matrix.append(row)
    return matrix


def _build_sparse_matrix(row_idx, duration, topics, topic_dict, keys=None):
    row_list = []
    col_list = []
    value_list = []

    row_list.append(row_idx)
    col_list.append(0)
    value_list.append(np.log10(int(duration)))

    topics = topics.split(',')
    for topic in topics:
        if keys is None or (keys is not None and topic in keys):
            row_list.append(row_idx)
            col_list.append(1 + topic_dict[topic])
            value_list.append(1)
    return row_list, col_list, value_list


def vectorize_train_data(data):
    train_row = []
    train_col = []
    train_value = []
    train_y = []
    topic_dict = {}
    topic_cnt = 0
    row_idx = 0

    for _, _, topics, _ in data:
        if not (topics == '' or topics == 'NA'):
            topics = topics.split(',')
            for topic in topics:
                if topic not in topic_dict:
                    topic_dict[topic] = topic_cnt
                    topic_cnt += 1

    for _, duration, topics, target in data:
        if not (topics == '' or topics == 'NA'):
            row_list, col_list, value_list = _build_sparse_matrix(row_idx, duration, topics, topic_dict)
            train_row.extend(row_list)
            train_col.extend(col_list)
            train_value.extend(value_list)
            train_y.append(float(target))
            row_idx += 1
    return coo_matrix((train_value, (train_row, train_col)), shape=(row_idx, topic_cnt+1)), train_y, topic_dict


def vectorize_test_data(data, topic_dict):
    test_vids = []
    test_row = []
    test_col = []
    test_value = []
    test_y = []
    n_topic = len(topic_dict)
    topic_keys = list(topic_dict.keys())
    row_idx = 0

    for vid, duration, topics, target in data:
        if not (topics == '' or topics == 'NA'):
            row_list, col_list, value_list = _build_sparse_matrix(row_idx, duration, topics, topic_dict, keys=topic_keys)
            test_row.extend(row_list)
            test_col.extend(col_list)
            test_value.extend(value_list)
            test_y.append(float(target))
            row_idx += 1
            test_vids.append(vid)
    return coo_matrix((test_value, (test_row, test_col)), shape=(row_idx, n_topic+1)), test_y, test_vids


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    start_time = time.time()

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
        print('>>> Start to predict watch percentage with freebase topics in sparse matrix...')
    elif target == 're':
        is_re = True
        print('>>> Start to predict relative engagement with freebase topics in sparse matrix...')
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

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

    # write to pickle file
    to_write = True
    predict_result_dict = {vid: pred for vid, pred in zip(test_vids, test_yhat)}
    if to_write:
        print('>>> Prepare to write to pickle file...')
        print('>>> Number of videos in final test result dict: {0}'.format(len(predict_result_dict)))
        write_dict_to_pickle(dict=predict_result_dict,
                             path=os.path.join(output_dir, '{0}_sparse_topic_predictor.p'.format(['wp', 're'][is_re])))
