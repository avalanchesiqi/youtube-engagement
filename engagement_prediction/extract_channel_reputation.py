#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract channel past reputation from training dataset.

Target: extract watch percentage
Usage: python extract_channel_reputation.py -i ./ -o ./output/train_channel_watch_percentage.txt -f wp
Time: ~1M

Target: extract relative engagement
Usage: python extract_channel_reputation.py -i ./ -o ./output/train_channel_relative_engagement.txt -f re
Time: ~1M
"""

import sys, os, time, datetime, argparse


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    start_time = time.time()

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file dir of training dataset', required=True)
    parser.add_argument('-o', '--output', help='output file path of channel past reputation', required=True)
    parser.add_argument('-f', '--function', help='choose prediction target', required=True)
    args = parser.parse_args()

    input_dir = args.input
    output_path = args.output
    target = args.function

    if target == 'wp':
        is_re = False
        print('>>> Start to extract channel past watch percentage from training dataset...')
    elif target == 're':
        is_re = True
        print('>>> Start to extract channel past relative engagement from training dataset...')
    else:
        print('>>> Error: Unknown prediction target! It mush be wp or re!')
        print('>>> Exit...')
        print(sys.exit(1))

    train_loc = os.path.join(input_dir, 'train_data')
    if not os.path.exists(train_loc):
        print('>>> Error: Did not find train dataset!')
        print('>>> Exit...')
        print(sys.exit(1))

    output_data = open(output_path, 'w')
    for subdir, _, files in os.walk(train_loc):
        for f in files:
            with open(os.path.join(subdir, f), 'r') as fin:
                fin.readline()
                for line in fin:
                    _, _, _, _, _, _, channel_id, _, _, _, wp30, re30, _ = line.rstrip().split('\t', 12)
                    target = [wp30, re30][is_re]
                    output_data.write('{0}\t{1}\n'.format(channel_id, target))
    output_data.close()

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])
