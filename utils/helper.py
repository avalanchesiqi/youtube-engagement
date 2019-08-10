import os, pickle, time
from datetime import timedelta
import numpy as np


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        print('>>> Elapsed time: {0}\n'.format(str(timedelta(seconds=time.time() - self.start_time))[:-3]))


def read_as_int_array(content, truncated=None, delimiter=None):
    """ Read input as an int array.
    :param content: string input
    :param truncated: head number of elements extracted
    :param delimiter: delimiter string
    :return: a numpy int array
    """
    if truncated is None:
        return np.array(list(map(int, content.split(delimiter))), dtype=np.uint32)
    else:
        return np.array(list(map(int, content.split(delimiter)[:truncated])), dtype=np.uint32)


def read_as_float_array(content, truncated=None, delimiter=None):
    """ Read input as a float array.
    :param content: string input
    :param truncated: head number of elements extracted
    :param delimiter: delimiter string
    :return: a numpy float array
    """
    if truncated is None:
        return np.array(list(map(float, content.split(delimiter))), dtype=np.float64)
    else:
        return np.array(list(map(float, content.split(delimiter)[:truncated])), dtype=np.float64)


def strify(iterable_struct, delimiter=','):
    """ Convert an iterable structure to comma separated string.
    :param iterable_struct: an iterable structure
    :param delimiter: separated character, default comma
    :return: a string with delimiter separated
    """
    return delimiter.join(map(str, iterable_struct))


def write_dict_to_pickle(dict, path):
    """ Write a dictionary object into pickle file.
    :param dict: a dictionary object
    :param path: output pickle file path
    :return: 
    """
    folder_path = os.path.dirname(path)
    if not (folder_path == '' or os.path.exists(folder_path)):
        os.makedirs(folder_path)
    pickle.dump(dict, open(path, 'wb'))


def exponent_fmt(x, pos):
    """ The two args are the value and tick position. """
    return '{0:.0f}'.format(10 ** x)
