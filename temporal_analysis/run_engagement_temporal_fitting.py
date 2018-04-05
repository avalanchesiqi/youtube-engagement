#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Scripts to run relative engagement temporal fitting.

Usage: python run_engagement_temporal_fitting.py -i ./sliding_engagement_dynamics.csv -o ./sliding_fitting_results.csv
Time: ~40M
"""

import sys, os, time, datetime, argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
from scipy.optimize import curve_fit

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import read_as_int_array, read_as_float_array


def func_powerlaw(x, a, b, c):
    return a * (x**b) + c


def fit_with_powerlaw(x, y, max_iter=10000):
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    try:
        params, _ = curve_fit(func_powerlaw, x, y, maxfev=max_iter)
        return params
    except Exception as e:
        print('+++ Fail to fit with power law with error message,', str(e))
        if 'has reached maxfev' in str(e):
            if max_iter == 50000:
                pass
            else:
                return fit_with_powerlaw(x, y, max_iter=50000)


if __name__ == '__main__':
    # == == == == == == == == Part 1: Set up experiment parameters == == == == == == == == #
    print('>>> Start to fit sliding window temporal engagement dynamics...')
    start_time = time.time()
    age = 30
    ts = np.arange(1, age + 1).reshape(-1, 1)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='input file path of sliding window relative engagement dynamics', required=True)
    parser.add_argument('-o', '--output', help='output file path of sliding window fitting results', required=True)
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    fitting_output = open(output_path, 'w')
    fitting_output.write('Vid,Err_Powerlaw,Err_linear,Err_constant\n')

    with open(input_path, 'r') as fin:
        for line in fin:
            vid, days, _, re_list = line.rstrip().split('\t')
            days = read_as_int_array(days, delimiter=',')
            re_list = read_as_float_array(re_list, delimiter=',')

            # pre-requisite: views in all sliding windows (first 30 days) are at least 100
            if len(days) == age:
                # power-law
                optimal_params = fit_with_powerlaw(days, re_list)
                if optimal_params is not None:
                    mae_powerlaw = mean_absolute_error(re_list, [func_powerlaw(x, *optimal_params) for x in days])

                    # linear reg
                    model = LinearRegression()
                    model.fit(ts, re_list)
                    mae_linear = mean_absolute_error(re_list, model.predict(ts))

                    # constant
                    mae_constant = mean_absolute_error(re_list, [np.median(re_list)]*age)

                    fitting_output.write('{0},{1},{2},{3}\n'.format(vid, mae_powerlaw, mae_linear, mae_constant))

    fitting_output.close()

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])

