#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to run relative engagement temporal fitting.

Usage: python run_engagement_temporal_fitting.py -i ./sliding_engagement_dynamics.csv
Time: 
"""

import sys, os, argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
from scipy.optimize import curve_fit

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import read_as_float_array


def func_powerlaw(x, a, b, c):
    return a * (x**b) + c


def fit_with_powerlaw(x, y):
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    try:
        params, _ = curve_fit(func_powerlaw, x, y, maxfev=10000)
        return params
    except:
        pass


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to fit sliding window temporal engagement dynamics...')
    age = 30
    ts = np.arange(1, age + 1).reshape(-1, 1)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file path of sliding window relative engagement dynamics', required=True)
    args = parser.parse_args()

    input_path = args.input
    fitting_output = open('./sliding_fitting_results.csv', 'w')

    with open(input_path, 'r') as fin:
        for line in fin:
            vid, _, _, re_list = line.rstrip().split('\t')
            re_list = read_as_float_array(re_list, delimiter=',')

            # pre-requisite: views in all sliding windows (first 30 days) are at least 100
            if len(re_list) == age:
                # power-law
                optimal_params = fit_with_powerlaw(ts, re_list)
                if optimal_params is not None:
                    error_power = mean_absolute_error(re_list, [func_powerlaw(x, *optimal_params) for x in ts])

                    # linear reg
                    model = LinearRegression()
                    model.fit(ts, re_list)
                    error_linear = mean_absolute_error(re_list, model.predict(ts))

                    # constant
                    error_constant = mean_absolute_error(re_list, [np.median(re_list)]*age)

                    fitting_output.write('{0},{1},{2},{3}\n'.format(vid, error_power, error_linear, error_constant))

    fitting_output.close()

