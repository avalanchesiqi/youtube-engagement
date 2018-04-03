#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Script to plot Figure 9, show conditional entropy for frequent topics.
Calculate conditional entropy between topic type and relative engagement, i.e, I(book; eta),
by constructing 2x20 oc-occurrence matrix
X                0     1
Y     0-0.05    10    15
   0.05-0.10    20    25
       ....
   0.95-1.00    30    35

I(X;Y) = sum(P(x, y) * log( P(x, y)/P(x)/P(y) ))

Usage: python plot_fig9_topic_conditional_entropy.py
Time: ~7M
"""

import os, sys, time, datetime, operator
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter


def exponent(x, pos):
    """ The two args are the value and tick position. """
    return '{0:.0f}'.format(x)


def rescale(x, pos):
    """ The two args are the value and tick position. """
    return '{0:1.1f}'.format(x / 20)


def safe_log2(x):
    if x == 0:
        return 0
    else:
        return np.log2(x)


def get_conditional_entropy(topic_eta):
    # calculate condition entropy when topic appears
    binned_topic_eta = {i: 0 for i in range(bin_num)}
    for eta in topic_eta:
        binned_topic_eta[min(int(eta / bin_gap), bin_num - 1)] += 1

    p_Y_given_x1 = [binned_topic_eta[i] / len(topic_eta) for i in range(bin_num)]
    return -np.sum([p * safe_log2(p) for p in p_Y_given_x1]), [binned_topic_eta[x] for x in sorted(binned_topic_eta.keys())]


def _load_data(filepath):
    with open(filepath, 'r') as fin:
        fin.readline()
        for line in fin:
            vid, _, _, _, _, _, _, topics, _, _, _, re30, _ = line.rstrip().split('\t', 12)
            if not (topics == '' or topics == 'NA'):
                topics = topics.split(',')
                re30 = float(re30)
                video_topic_set = set()
                for topic in topics:
                    if topic in mid_type_dict:
                        freebase_types = mid_type_dict[topic].split(',')
                        for ft in freebase_types:
                            if ft != 'common' and ft != 'type_ontology' and ft != 'type':
                                video_topic_set.add(ft)

                for ft in video_topic_set:
                    type_eta_dict[ft].append(re30)
                    type_eta_counter_dict[ft] += 1


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to plot conditional entropy for frequent topics...')
    start_time = time.time()

    bin_gap = 0.05
    bin_num = int(1 / bin_gap)

    # == == == == == == == == Part 2: Load Freebase dictionary == == == == == == == == #
    freebase_dict_loc = '../data/freebase_mid_type_name.csv'
    if not os.path.exists(freebase_dict_loc):
        print('Freebase dictionary not found!')
        print('Exit program...')
        sys.exit(1)

    mid_type_dict = {}
    with open(freebase_dict_loc, 'r', encoding='utf8') as fin:
        for line in fin:
            mid, type, _ = line.rstrip().split('\t', 2)
            mid_type_dict[mid] = type

    # == == == == == == == == Part 3: Load dataset == == == == == == == == #
    train_dataset_path = '../engagement_prediction/train_data'
    if not os.path.exists(train_dataset_path):
        print('>>> No train data found!')
        print('Exit program...')
        sys.exit(1)

    type_eta_dict = defaultdict(list)
    type_eta_counter_dict = defaultdict(int)
    print('>>> Start to load TWEETED VIDEOS dataset...')
    for subdir, _, files in os.walk(train_dataset_path):
        for f in files:
            print('>>> Start to load file: {0}...'.format(os.path.join(subdir, f)))
            _load_data(os.path.join(subdir, f))
    print('>>> Finish loading all data!')
    print('>> Number of topic types: {0}\n'.format(len(type_eta_dict)))

    # == == == == == Part 4: Calculate conditional entropy for topic type and relative engagement == == == == == #
    sorted_type_eta_counter = sorted(type_eta_counter_dict.items(), key=operator.itemgetter(1), reverse=True)[:500]
    print('>>> Largest 500 clusters:')
    print(sorted_type_eta_counter)

    type_conditional_entropy_dict = {}
    print('>>> Show a list of informative topics:')
    print('>>> Type name | Number of appearance | Conditional entropy | Mean relative engagement')
    for type, _ in sorted_type_eta_counter:
        # type size, conditional entropy, mean eta value
        conditional_entropy = get_conditional_entropy(type_eta_dict[type])[0]
        type_conditional_entropy_dict[type] = (type_eta_counter_dict[type], conditional_entropy, np.mean(type_eta_dict[type]))
        # print when condition entropy is small
        if conditional_entropy < 4.1:
            print(type, type_conditional_entropy_dict[type])

    # == == == == == == == == Part 5: Generate bar plots == == == == == == == == #
    to_plot = True
    if to_plot:
        fig = plt.figure(figsize=(8, 5))
        cornflower_blue = (0.3921, 0.5843, 0.9294)
        tomato = (1.0, 0.3882, 0.2784)

        def make_colormap(seq):
            """ Return a LinearSegmentedColormap
            seq: a sequence of floats and RGB-tuples. The floats should be increasing
            and in the interval (0,1).
            """
            seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
            cdict = {'red': [], 'green': [], 'blue': []}
            for i, item in enumerate(seq):
                if isinstance(item, float):
                    r1, g1, b1 = seq[i - 1]
                    r2, g2, b2 = seq[i + 1]
                    cdict['red'].append([item, r1, r2])
                    cdict['green'].append([item, g1, g2])
                    cdict['blue'].append([item, b1, b2])
            return mcolors.LinearSegmentedColormap('CustomMap', cdict)

        c = mcolors.ColorConverter().to_rgb
        rvb = make_colormap([cornflower_blue, c('white'), 0.5, c('white'), tomato])

        keys = type_conditional_entropy_dict.keys()
        x_axis = [type_conditional_entropy_dict[x][0] for x in keys]
        y_axis = [type_conditional_entropy_dict[x][1] for x in keys]
        colors = [type_conditional_entropy_dict[x][2] for x in keys]
        plt.scatter(x_axis, y_axis, c=colors, edgecolors='none', cmap=rvb)

        # plot annotated topics
        plt.text(type_conditional_entropy_dict['baseball'][0], type_conditional_entropy_dict['baseball'][1], 'baseball', size=16, ha='left', va='top')
        plt.text(type_conditional_entropy_dict['book'][0], type_conditional_entropy_dict['book'][1], 'book', size=16, ha='center', va='top')
        plt.text(type_conditional_entropy_dict['music'][0], type_conditional_entropy_dict['music'][1], 'music', size=16, ha='center', va='top')
        plt.text(type_conditional_entropy_dict['bollywood'][0], type_conditional_entropy_dict['bollywood'][1], 'bollywood', size=16, ha='left', va='bottom')
        plt.text(type_conditional_entropy_dict['obamabase'][0], type_conditional_entropy_dict['obamabase'][1], 'obama', size=16, ha='left', va='bottom')

        plt.xscale('log')
        plt.gca().xaxis.set_major_formatter(FuncFormatter(exponent))
        plt.ylim(ymax=-np.sum([p * safe_log2(p) for p in [bin_gap]*bin_num]))
        plt.ylim(ymin=3.88)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('topic size', fontsize=16)
        plt.ylabel('conditional entropy', fontsize=16)

        cb = plt.colorbar()
        cb.set_label(label='relative engagement $\eta$', size=16)
        cb.ax.tick_params(labelsize=14)

        # inset subfigure
        ax2 = fig.add_axes([0.42, 0.24, 0.35, 0.35])
        width = 1 / 2
        ind = np.arange(20)

        count_freq1 = get_conditional_entropy(type_eta_dict['bollywood'])[1]
        prob1 = [x / np.sum(count_freq1) for x in count_freq1]
        ax2.bar(ind + width * 1 / 2, prob1, width, color=cb.to_rgba(type_conditional_entropy_dict['bollywood'][2]), label='bollywood')

        count_freq2 = get_conditional_entropy(type_eta_dict['obamabase'])[1]
        prob2 = [x / np.sum(count_freq2) for x in count_freq2]
        ax2.bar(ind + width * 3 / 2, prob2, width, color=cb.to_rgba(type_conditional_entropy_dict['obamabase'][2]), label='obama')

        ax2.set_xlim([0, 20])
        ax2.set_ylim([0, 0.15])
        ax2.set_xticks([0, 4, 8, 12, 16, 20])

        ax2.xaxis.set_major_formatter(FuncFormatter(rescale))
        ax2.set_xlabel('relative engagement $\eta$', fontsize=8)
        ax2.set_ylabel('engagement probability', fontsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax2.set_yticks(ax2.get_yticks()[::2])
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.legend(loc='upper right', fontsize=12, frameon=False)

        # get running time
        print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

        plt.tight_layout()
        plt.savefig('../images/fig9_topics_conditional_entropy.pdf', bbox_inches='tight')
        plt.show()
