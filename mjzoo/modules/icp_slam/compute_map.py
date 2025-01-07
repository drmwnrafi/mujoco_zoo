from skimage.draw import line as draw_line
import numpy as np

def log_odds(prob):
  return np.log(prob / (1 - prob))

def retrieve_prob(log_odds):
  return 1 - 1 /(1 + np.exp(log_odds))

def max_likelihood(map, free_threshold, occupied_threshold):
    log_odds_map = np.full_like(map, log_odds(0.5))
    np.place(log_odds_map, map < free_threshold, log_odds(0.01))
    np.place(log_odds_map, map > occupied_threshold, log_odds(0.99))
    return log_odds_map

class GridMap:
    def __init__(self, resolution, prob_free, prob_occ, map_offset=5):
        self.prob_free = log_odds(prob_free)
        self.prob_occ = log_odds(prob_occ)
        self.TRESHOLD_P_FREE = 0.3
        self.TRESHOLD_P_OCC = 0.6
        self.resolution = resolution
        self.offset = map_offset
        self.map = None

    def update(self, pose, points):
        zero = (pose // self.resolution).astype(np.int32)
        pixels = (points // self.resolution).astype(np.int32)

        for x, y in pixels:
            rr, cc = draw_line(zero[0], zero[1], x, y)
            self.map[rr, cc] += self.prob_free
            self.map[x, y] += self.prob_occ

    def prob_to_map(self):
        return retrieve_prob(self.map).T[::-1]

    def mle_map(self):
        return max_likelihood(self.map, log_odds(self.TRESHOLD_P_FREE), log_odds(self.TRESHOLD_P_OCC)).T[::-1]
        
    def clear_map(self, map_size):
        map_size = (map_size[0] + self.offset, map_size[1] + self.offset)
        self.map = np.full(map_size, fill_value=log_odds(0.5))