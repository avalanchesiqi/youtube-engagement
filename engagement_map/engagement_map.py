import os, pickle


class EngagementMap(object):
    def __init__(self, engagement_path):
        self.engagement_map = None
        self.duration_axis = None
        self.bin_size = None
        self.load_engagement_map(engagement_path)

    def load_engagement_map(self, engagement_filepath):
        """ Load engagement map.
        """
        if not os.path.exists(engagement_filepath):
            raise Exception('no engagement file is found!')

        self.engagement_map = pickle.load(open(engagement_filepath, 'rb'))
        self.duration_axis = self.engagement_map['duration']
        self.bin_size = len(self.engagement_map[self.engagement_map['duration'][0]])

    def query_engagement_map(self, duration, watch_percentage):
        """ Query the engagement map for relative engagement given video length and watch percentage.
        """
        try:
            bin_x_idx = next(idx for idx, length in enumerate(self.duration_axis) if length >= duration)
        except StopIteration:
            bin_x_idx = len(self.duration_axis) - 1

        correspond_watch_percentage = self.engagement_map[bin_x_idx]
        try:
            relative_engagement = next(y for y, val in enumerate(correspond_watch_percentage) if val > watch_percentage) / self.bin_size
        except StopIteration:
            relative_engagement = 1
        return relative_engagement
