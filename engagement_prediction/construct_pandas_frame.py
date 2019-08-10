#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Construct pandas dataframe from predictor pickle output.

Watch percentage example rows
           Vid        True    Duration     Context       Topic      CTopic  Reputation         All         CSP
0  KcgjkCDPOco    0.300155    0.300902    0.343142    0.348659    0.378310    0.589099    0.608004    0.608004
1  oydbUUFZNPQ    0.339141    0.424607    0.392023    1.000000    0.501331    0.431703    0.472517    0.438917
2  RUAKJSxfgW0    0.694815    0.454994    0.473313    0.587380    0.586567    0.582470    0.610346    0.526601
3  U45p1d_zQEs    0.491463    0.486584    0.514190    0.502563    0.508860    0.602810    0.554918    0.623619
4  wjdjztvb9Hc    0.832867    0.541110    0.556331    1.000000    0.501331    0.781969    0.472517    0.472517

Relative engagement example rows
           Vid     True    Duration     Context       Topic      CTopic  Reputation          All         CSP  VDuration
0  KcgjkCDPOco    0.498         0.5    0.608499    0.610826    0.667708    0.871146     0.878522    0.878522       1212
1  oydbUUFZNPQ    0.301         0.5    0.405562    0.635107    0.515533    0.424186     0.489899    0.435074        350
2  RUAKJSxfgW0    0.945         0.5    0.462427    0.737947    0.738719    0.699488     0.715835    0.450224        254
3  U45p1d_zQEs    0.512         0.5    0.504788    0.491619    0.501147    0.127934     0.142407    0.000000        209
4  wjdjztvb9Hc    0.988         0.5    0.523769    0.635107    0.515533    0.994331     0.489899    0.489899        160

Target: merge watch percentage
Usage: python construct_pandas_frame.py -i ./output -o ./output/predicted_wp_df.csv -f wp
Time: ~5M

Target: merge relative engagement
Usage: python construct_pandas_frame.py -i ./output -o ./output/predicted_re_df.csv -f re
Time: ~5M
"""

import sys, os, pickle, argparse
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    timer = Timer()
    timer.start()

    # == == == == == == == == Part 2: Load dataset == == == == == == == == #
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file dir of predicted pickles', required=True)
    parser.add_argument('-o', '--output', help='output file path of merged pandas dataframe', required=True)
    parser.add_argument('-f', '--function', help='choose merge target', required=True)
    args = parser.parse_args()

    input_dir = args.input
    output_path = args.output
    target = args.function

    if target == 'wp':
        is_re = False
        print('>>> Start to build pandas dataframe for watch percentage...')
    elif target == 're':
        is_re = True
        print('>>> Start to build pandas dataframe for relative engagement...')
    else:
        print('>>> Error: Unknown prediction target! It mush be wp or re!')
        print('>>> Exit...')
        print(sys.exit(1))

    # construct pandas dataframe if not exists
    if not os.path.exists(output_path):
        true_dict_path = os.path.join(input_dir, '{0}_true_predictor.p'.format(['wp', 're'][is_re]))
        duration_predictor_path = os.path.join(input_dir, '{0}_duration_predictor.p'.format(['wp', 're'][is_re]))
        context_predictor_path = os.path.join(input_dir, '{0}_context_predictor.p'.format(['wp', 're'][is_re]))
        topic_predictor_path = os.path.join(input_dir, '{0}_sparse_topic_predictor.p'.format(['wp', 're'][is_re]))
        context_topic_predictor_path = os.path.join(input_dir, '{0}_sparse_context_topic_predictor.p'.format(['wp', 're'][is_re]))
        channel_reputation_predictor_path = os.path.join(input_dir, '{0}_reputation_predictor.p'.format(['wp', 're'][is_re]))
        all_predictor_path = os.path.join(input_dir, '{0}_sparse_all_predictor.p'.format(['wp', 're'][is_re]))
        channel_specific_predictor_path = os.path.join(input_dir, '{0}_csp_predictor_5.p'.format(['wp', 're'][is_re]))
        test_duration_path = os.path.join(input_dir, 'test_duration.p')

        # ground-truth values
        true_dict = pickle.load(open(true_dict_path, 'rb'))
        vids = true_dict.keys()

        # duration predictor
        duration_predictor = pickle.load(open(duration_predictor_path, 'rb'))

        # context predictor
        context_predictor = pickle.load(open(context_predictor_path, 'rb'))

        # topic predictor
        topic_predictor = pickle.load(open(topic_predictor_path, 'rb'))
        for vid in vids:
            if vid not in topic_predictor:
                topic_predictor[vid] = duration_predictor[vid]

        # context topic predictor
        context_topic_predictor = pickle.load(open(context_topic_predictor_path, 'rb'))
        for vid in vids:
            if vid not in context_topic_predictor:
                context_topic_predictor[vid] = context_predictor[vid]

        # channel reputation predictor
        channel_reputation_predictor = pickle.load(open(channel_reputation_predictor_path, 'rb'))
        for vid in vids:
            if vid not in channel_reputation_predictor:
                channel_reputation_predictor[vid] = duration_predictor[vid]

        # all features predictor
        all_predictor = pickle.load(open(all_predictor_path, 'rb'))
        for vid in vids:
            if vid not in all_predictor:
                if not channel_reputation_predictor[vid] == duration_predictor[vid]:
                    all_predictor[vid] = channel_reputation_predictor[vid]
                else:
                    all_predictor[vid] = context_topic_predictor[vid]

        # channel specific predictor
        channel_specific_predictor = pickle.load(open(channel_specific_predictor_path, 'rb'))
        for vid in vids:
            if vid not in channel_specific_predictor:
                channel_specific_predictor[vid] = all_predictor[vid]

        # test duration
        test_duration = pickle.load(open(test_duration_path, 'rb'))

        # generate pandas dataframe
        true_data_f = pd.DataFrame(list(true_dict.items()), columns=['Vid', 'True'])
        duration_data_f = pd.DataFrame(list(duration_predictor.items()), columns=['Vid', 'Duration'])
        context_data_f = pd.DataFrame(list(context_predictor.items()), columns=['Vid', 'Context'])
        topic_data_f = pd.DataFrame(list(topic_predictor.items()), columns=['Vid', 'Topic'])
        context_topic_data_f = pd.DataFrame(list(context_topic_predictor.items()), columns=['Vid', 'CTopic'])
        channel_reputation_data_f = pd.DataFrame(list(channel_reputation_predictor.items()), columns=['Vid', 'Reputation'])
        all_data_f = pd.DataFrame(list(all_predictor.items()), columns=['Vid', 'All'])
        channel_specific_data_f = pd.DataFrame(list(channel_specific_predictor.items()), columns=['Vid', 'CSP'])
        test_duration_data_f = pd.DataFrame(list(test_duration.items()), columns=['Vid', 'VDuration'])
        data_f = true_data_f.merge(duration_data_f, on='Vid')\
            .merge(context_data_f, on='Vid')\
            .merge(topic_data_f, on='Vid')\
            .merge(context_topic_data_f, on='Vid')\
            .merge(channel_reputation_data_f, on='Vid')\
            .merge(all_data_f, on='Vid')\
            .merge(channel_specific_data_f, on='Vid')\
            .merge(test_duration_data_f, on='Vid')

        for name in ['True', 'Duration', 'Context', 'Topic', 'CTopic', 'Reputation', 'All', 'CSP']:
            data_f[name] = data_f[name].where(data_f[name] < 1, 1)
            data_f[name] = data_f[name].where(data_f[name] > 0, 0)
        data_f.to_csv(output_path, sep='\t')

        print('header:')
        print(data_f.head())

    timer.stop()
