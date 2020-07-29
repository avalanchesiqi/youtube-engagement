import random
from engagement_map import EngagementMap

engagement_path = '../data/engagement_map.p'
engagement_map = EngagementMap(engagement_path)

for _ in range(10):
    length = 10 ** (5 * random.random())
    wp30 = random.random()
    print('relative engagement for video with length {0:.0f} seconds and {1:.2f} watch percentage is {2:.2f}'.format(length, wp30, engagement_map.query_engagement_map(length, wp30)))
