#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to construct formatted data from raw json collection.
Note the released dataset is cleaned, i.e., all invalid records are erased. If one starts the data collection/analysis
from scratch, s/he may need to apply additional filters and handle corrupted data.

Usage: python construct_formatted_data.py -i input_doc -o output_doc
Example: python construct_formatted_dataset.py -i ../data/tweeted_videos -o ../data/formatted_tweeted_videos
Time: ~20H20M
Example: python construct_formatted_dataset.py -i ../data/quality_videos -o ../data/formatted_quality_videos
Time: ~2M
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import argparse, json, isodate, time
from datetime import datetime, timedelta
import numpy as np

from utils.helper import read_as_float_array, read_as_int_array, strify


def extract_info(input_path, output_path, truncated=None):
    """
    Extract essential information from each video.
    :param input_path: input file path
    :param output_path: output file path
    :param truncated: head number of extracted elements in attention dynamics
    :return:
    """
    fout = open(output_path, 'w')
    fout.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\n'
               .format('id', 'publish', 'duration', 'definition', 'category', 'detect_lang', 'channel', 'topics',
                       'view30', 'watch30', 'wp30', 'days', 'daily_view', 'daily_watch'))

    with open(input_path, 'r') as fin:
        for line in fin:
            # skip if data is corrupted or reading duration fails
            try:
                video = json.loads(line.rstrip())
            except:
                continue

            vid = video['id']
            published_at = video['snippet']['publishedAt'][:10]
            duration = isodate.parse_duration(video['contentDetails']['duration']).seconds
            definition = [0, 1][video['contentDetails']['definition'] == 'hd']
            category = video['snippet']['categoryId']
            detect_lang = video['snippet']['detectLang']
            channel = video['snippet']['channelId']

            # freebase topic information
            if 'topicDetails' in video:
                if 'topicIds' in video['topicDetails']:
                    topic_ids = set(video['topicDetails']['topicIds'])
                else:
                    topic_ids = set()
                if 'relevantTopicIds' in video['topicDetails']:
                    relevant_topic_ids = set(video['topicDetails']['relevantTopicIds'])
                else:
                    relevant_topic_ids = set()
                topics_set = topic_ids.union(relevant_topic_ids)
                topics = strify(topics_set)
            else:
                topics = 'NA'

            # attention dynamics information
            start_date = video['insights']['startDate']
            time_diff = (datetime(*map(int, start_date.split('-'))) - datetime(*map(int, published_at.split('-')))).days
            days = read_as_int_array(video['insights']['days'], delimiter=',', truncated=truncated) + time_diff
            days = days[days < truncated]
            daily_view = read_as_int_array(video['insights']['dailyView'], delimiter=',', truncated=len(days))
            view30 = np.sum(daily_view[days < 30])
            daily_watch = read_as_float_array(video['insights']['dailyWatch'], delimiter=',', truncated=len(days))
            watch30 = np.sum(daily_watch[days < 30])
            # I have cleaned the data, so views in the first 30 days will be greater than 100
            # take care of zero view and very occasionally (streamed video) zero duration
            wp30 = watch30*60/view30/duration
            # upper bound watch percentage to 1
            if wp30 > 1:
                wp30 = 1

            fout.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\n'
                       .format(vid, published_at, duration, definition, category, detect_lang, channel, topics,
                               view30, watch30, wp30, strify(days), strify(daily_view), strify(daily_watch)))
    fout.close()


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    # setting parameters
    print('>>> Start to convert video raw json file to formatted text file...')
    start_time = time.time()
    age = 120

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file dir of raw json collection', required=True)
    parser.add_argument('-o', '--output', help='output file dir of formatted data', required=True)
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    if not os.path.exists(input_dir):
        print('>>> Input file dir does not exist!')
        print('>>> Exit...')
        sys.exit(1)

    if os.path.exists(output_dir):
        print('>>> Output file dir already exists, rename, check or backup it before starting new job!')
        print('>>> Exit...')
        sys.exit(1)
    else:
        os.mkdir(output_dir)

    # == == == == == == == == Part 2: Construct dataset == == == == == == == == #
    for subdir, _, files in os.walk(input_dir):
        for f in files:
            extract_info(os.path.join(subdir, f), os.path.join(output_dir, f[:-4]+'txt'), truncated=age)
            print('>>> Finish extracting file {0}!'.format(os.path.join(subdir, f)))

    # get running time
    print('\n>>> Total running time: {0}'.format(str(timedelta(seconds=time.time() - start_time)))[:-3])
