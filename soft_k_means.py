# -*- coding:utf-8 -*-

import minpy.numpy as np
from math import sin
from math import cos

def logsumexp(data, axis):
    with np.errstate(under='ignore'):
        data = np.log(np.sum(np.exp(data), axis=axis))
    return data


class SoftKMeans2d(object):
    means_ = None

    def __init__(self, n_components,
                 init_sigma,
                 max_iter,
                 x_min,
                 x_max,
                 y_min,
                 y_max):
        self.n_components = n_components
        self.sigma = init_sigma
        self.max_iter = max_iter
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def hard_k_means_fit(self,
                         data,
                         x_min, x_max,
                         y_min, y_max,
                         iter_num=300):
        n_samples, n_features = data.shape
        x_means = np.random.uniform(x_min, x_max, (self.n_components, 1))
        y_means = np.random.uniform(y_min, y_max, (self.n_components, 1))
        self.means_ = np.concatenate([x_means, y_means], axis=1)

        for i in range(iter_num):
            distance_vector = (np.stack(data.T, axis=0) - np.stack(self.means_, axis=2))
            distance = np.sqrt(np.square(distance_vector[:, 0, :])
                               + np.square(distance_vector[:, 1, :]))
            means_index = np.argmin(distance, axis=0)
            resp = np.zeros((self.n_components, n_samples))
            resp[means_index, np.arange(0, n_samples, 1)] = 1
            resp /= np.stack(np.sum(resp, axis=1), axis=1)
            self.means_ = np.dot(resp, data)

    def fit(self, data,
            x_min,
            x_max,
            y_min,
            y_max,
            kmeans=True):
        """
        训练
        :param data: 数据（n_samples * n_features）
        :param x_min:
        :param x_max:
        :param y_min:
        :param y_max: 设定一块区域
        :param kmeans: 是否通过hard_k_means进行初始化
        :return: n_components * n_features
        """
        n_samples, n_features = data.shape
        if kmeans:
            self.hard_k_means_fit(data, x_min, x_max, y_min, y_max)
        else:
            x_means = np.random.uniform(x_min, x_max, (1, n_features))
            y_means = np.random.uniform(y_min, y_max, (1, n_features))
            self.means_ = np.concatenate([x_means, y_means], axis=0)

        for n_iter in range(self.max_iter):
            log_prob_norm, log_resp = self._e_step(data)
            self._m_step(data, log_resp)
            self.set_mean_nan_to_random()
            means_distance = self.means_distance()
            means_distance_logic = self.means_distance_logic(means_distance, 0.5)
            near_distance_means_dict = self.near_distance_means_dict(means_distance_logic)
            print near_distance_means_dict
            self.avoid_near_mean(near_distance_means_dict)
        return self

    def avoid_near_mean(self, near_distance_means_dict):
        # beta = (self.x_max - self.x_min) * (self.y_max - self.y_min) / self.n_components
        for index in near_distance_means_dict:
            '''
            if np.size(near_distance_means_dict[index]) > 1:
                near_list = near_distance_means_dict[index]
                mean_x = np.mean(self.means_[near_list], [0])
                mean_y = np.mean(self.means_[near_list], [1])
                for i in near_list:
                    pi_r_2 = np.random.exponential(beta)
                    theta = np.random.uniform(0, 2 * np.pi)
                    r = pi_r_2 / np.pi ** (1/2)
                    self.means_[i, 0] = mean_x[0] + r * cos(theta[0])
                    self.means_[i, 1] = mean_y[0] + r * sin(theta[0])

            '''
            if np.size(near_distance_means_dict[index]) > 1:
                near_list = near_distance_means_dict[index]
                for mean_index in near_list:
                    self.means_[mean_index, 0] = np.random.uniform(self.x_min,
                                                                   self.x_max)
                    self.means_[mean_index, 1] = np.random.uniform(self.y_min,
                                                                   self.y_max)

                else:
                    continue
        return self

    visited_all = np.array([], dtype=np.int)

    def near_distance_means_dict(self, means_distance_logic):
        near_distance_means_dict = {}
        clusters = 0
        for i in range(0, self.n_components):
            if i not in self.visited_all:
                self.dfs(means_distance_logic, i)
                near_distance_means_dict[clusters] = self.visited_node
                self.visited_all = np.append(self.visited_all, self.visited_node)
                self.visited_node = np.array([], dtype=np.int)
                clusters += 1
            else:
                continue
        self.visited_all = np.array([])
        return near_distance_means_dict

    visited_node = np.array([], dtype=np.int)

    def dfs(self, means_distance_logic, node):
        if node not in self.visited_node:
            self.visited_node = np.append(self.visited_node, node)
        for i in range(self.n_components):

            if i not in self.visited_node\
                    and i != node\
                    and means_distance_logic[node, i] == 1:

                np.append(self.visited_node, i)
                self.dfs(means_distance_logic, i)
            else:
                continue

    def means_distance_logic(self, means_distance, avoid_distance):
        means_distance_logic = np.zeros(np.shape(means_distance), dtype=np.int)
        means_distance_logic = means_distance_logic.asnumpy()
        means_distance_logic[means_distance.asnumpy() <= avoid_distance] = 1
        means_distance_logic[means_distance.asnumpy() > avoid_distance] = 0
        means_distance_logic = np.array(means_distance_logic)
        return means_distance_logic

    def means_distance(self):
        distance_vector = np.stack(self.means_, axis=2) - np.stack(self.means_.T, axis=0)
        distance = np.sqrt(np.sum(np.square(distance_vector), axis=1))
        return distance

    def set_mean_nan_to_random(self):
        np.random.seed(np.random.randint(0, np.iinfo(np.uint32).max))
        means_x = self.means_[:, 0]
        index_nan_x = np.argwhere(np.isnan(means_x))
        for i in index_nan_x:
            means_x[i] = np.random.uniform(self.x_min, self.x_max)
        means_x = np.stack(means_x, axis=1)
        means_y = self.means_[:, 1]
        index_nan_y = np.argwhere(np.isnan(means_y))
        for i in index_nan_y:
            means_y[i] = np.random.uniform(self.y_min, self.y_max)
        means_y = np.stack(means_y, axis=1)
        self.means_ = np.concatenate([means_x, means_y], axis=1)

    def _e_step(self, data):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(data)
        return np.mean(log_prob_norm), log_resp

    def _estimate_log_prob_resp(self, data):
        weighted_log_prob = self._estimate_weighted_log_prob(data)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - np.stack(log_prob_norm, axis=1)
        return log_prob_norm, log_resp

    def _estimate_weighted_log_prob(self, data):
        log_prob = self._estimate_log_prob(data)
        log_weights = self._estimate_log_weights(log_prob)
        with np.errstate(under='ignore'):
            weighted_log_prob = log_prob + log_weights
        return weighted_log_prob

    def _estimate_log_prob(self, data):
        with np.errstate(under='ignore'):
            distance_vector = (np.stack(data.T, axis=0) - np.stack(self.means_, axis=2))
            distance = (np.square(distance_vector[:, 0, :])
                        + np.square(distance_vector[:, 1, :]))
            log_prob = - np.log(2 * np.pi * self.sigma) - distance / np.square(self.sigma)
        return log_prob

    def _estimate_log_weights(self, log_prob):
        # with np.errstate(under='ignore'):
        log_weights_unnorm = logsumexp(log_prob, axis=1)
        log_weights_unnorm = np.stack(log_weights_unnorm, axis=1)
        nan_index = np.argwhere(np.isnan(log_weights_unnorm))
        log_weights_unnorm[nan_index] = -np.inf
        norm_factor = logsumexp(log_weights_unnorm, axis=0)
        log_weights = log_weights_unnorm - norm_factor
        return log_weights

    def _m_step(self, data, log_resp):
        self.means_ = np.dot(np.exp(log_resp), data)
        return self

