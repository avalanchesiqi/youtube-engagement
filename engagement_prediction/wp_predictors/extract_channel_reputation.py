#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract channel past reputation from training dataset.

Usage: python extract_channel_reputation.py -i ../ -o ./output/train_channel_watch_percentage.txt
Time: ~3M
"""

import os, time, datetime, argparse


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to extract channel past reputation from training dataset...')
    start_time = time.time()

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file dir of training dataset', required=True)
    parser.add_argument('-o', '--output', help='output file path of channel past reputation', required=True)
    args = parser.parse_args()

    input_dir = args.input
    output_path = args.output
    train_loc = os.path.join(input_dir, 'train_data')
    output_data = open(output_path, 'w')

    for subdir, _, files in os.walk(train_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                fin.readline()
                for line in fin:
                    _, _, _, _, _, _, channel_id, _, _, _, wp30, _, _ = line.rstrip().split('\t', 12)
                    output_data.write('{0}\t{1}\n'.format(channel_id, wp30))

    output_data.close()

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])
