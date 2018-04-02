#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Construct pandas dataframe from predictor pickle output.

Example rows
           Vid     True    Duration     Context       Topic      CTopic  Reputation          All         CSP  VDuration
0  KcgjkCDPOco    0.498         0.5    0.608499    0.610826    0.667708    0.871146     0.878522    0.878522       1212
1  oydbUUFZNPQ    0.301         0.5    0.405562    0.635107    0.515533    0.424186     0.489899    0.435074        350
2  RUAKJSxfgW0    0.945         0.5    0.462427    0.737947    0.738719    0.699488     0.715835    0.450224        254
3  U45p1d_zQEs    0.512         0.5    0.504788    0.491619    0.501147    0.127934     0.142407    0.000000        209
4  wjdjztvb9Hc    0.988         0.5    0.523769    0.635107    0.515533    0.994331     0.489899    0.489899        160
"""

import os, pickle
import pandas as pd


if __name__ == '__main__':
    # construct pandas dataframe if not exists
    prefix_dir = './output/'
    dataframe_path = os.path.join(prefix_dir, 'predicted_re_sparse_df.csv')
    if not os.path.exists(dataframe_path):
        true_dict_path = os.path.join(prefix_dir, 'true_predictor.p')
        duration_predictor_path = os.path.join(prefix_dir, 'duration_predictor.p')
        context_predictor_path = os.path.join(prefix_dir, 'context_predictor.p')
        topic_predictor_path = os.path.join(prefix_dir, 'sparse_topic_predictor.p')
        context_topic_predictor_path = os.path.join(prefix_dir, 'sparse_context_topic_predictor.p')
        channel_reputation_predictor_path = os.path.join(prefix_dir, 'reputation_predictor.p')
        all_predictor_path = os.path.join(prefix_dir, 'sparse_all_predictor.p')
        channel_specific_predictor_path = os.path.join(prefix_dir, 'csp_predictor_5.p')
        test_duration_path = os.path.join(prefix_dir, 'test_duration.p')

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
                topic_predictor[vid] = 0.5

        # context topic predictor
        context_topic_predictor = pickle.load(open(context_topic_predictor_path, 'rb'))
        for vid in vids:
            if vid not in context_topic_predictor:
                context_topic_predictor[vid] = context_predictor[vid]

        # channel reputation predictor
        channel_reputation_predictor = pickle.load(open(channel_reputation_predictor_path, 'rb'))
        for vid in vids:
            if vid not in channel_reputation_predictor:
                channel_reputation_predictor[vid] = 0.5

        # all features predictor
        all_predictor = pickle.load(open(all_predictor_path, 'rb'))
        for vid in vids:
            if vid not in all_predictor:
                if not channel_reputation_predictor[vid] == 0.5:
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
        data_f = true_data_f.merge(duration_data_f, on='Vid') \
            .merge(context_data_f, on='Vid') \
            .merge(topic_data_f, on='Vid') \
            .merge(context_topic_data_f, on='Vid') \
            .merge(channel_reputation_data_f, on='Vid') \
            .merge(all_data_f, on='Vid') \
            .merge(channel_specific_data_f, on='Vid')\
            .merge(test_duration_data_f, on='Vid')

        for name in ['True', 'Duration', 'Context', 'Topic', 'CTopic', 'Reputation', 'All', 'CSP']:
            data_f[name] = data_f[name].where(data_f[name] < 1, 1)
            data_f[name] = data_f[name].where(data_f[name] > 0, 0)
        data_f.to_csv(dataframe_path, sep='\t')

        print('header:')
        print(data_f.head())
