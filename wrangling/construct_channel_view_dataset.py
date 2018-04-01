#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to construct train/test dataset in a channel view.

Usage: python construct_channel_view_dataset.py -i ../engagement_prediction
Time: ~20M
"""

import os, time, datetime, argparse


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to extract videos from one channel into one file...')
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file dir of train/test dataset in a category view', required=True)
    args = parser.parse_args()

    input_dir = args.input
    train_loc = os.path.join(input_dir, 'train_data')
    test_loc = os.path.join(input_dir, 'test_data')
    train_channel_view_loc = os.path.join(input_dir, 'train_data_channel_view')
    test_channel_view_loc = os.path.join(input_dir, 'test_data_channel_view')

    if not os.path.exists(train_channel_view_loc):
        os.mkdir(train_channel_view_loc)
    if not os.path.exists(test_channel_view_loc):
        os.mkdir(test_channel_view_loc)

    # == == == == == == == == Part 2: Construct channel view dataset == == == == == == == == #
    for input_loc, output_loc in [(train_loc, train_channel_view_loc), (test_loc, test_channel_view_loc)]:
        for subdir, _, files in os.walk(input_loc):
            for f in files:
                print('>>> Start to convert file {0}!'.format(os.path.join(subdir, f)))
                with open(os.path.join(subdir, f), 'r') as fin:
                    fin.readline()
                    for line in fin:
                        channel_id = line.rstrip().split('\t')[6]
                        sub_f = channel_id[:4]
                        if not os.path.exists(os.path.join(output_loc, sub_f)):
                            os.makedirs(os.path.join(output_loc, sub_f))
                        with open(os.path.join(output_loc, sub_f, channel_id), 'a') as fout:
                            fout.write(line)

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])