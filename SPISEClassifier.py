import random
import math
import numpy as np
from copy import deepcopy

from infinite_selection import InfFS
from scipy.stats import spearmanr


class SPISENovel(object):
    def __init__(self, base_estimator=None, n_estimators=10, n_subsets=3, n_entries=3):
        self.base_estimator = deepcopy(base_estimator)
        self.n_estimators = n_estimators
        self.n_subsets = n_subsets
        self.estimators = {}
        self.mappers = {}
        self.selector = {}
        self.reverse = False
        self.multi_class = False
        self.n_entries = n_entries

    def fit(self, X, y):
        y_max = max(y)
        y_min = min(y)
        n_candidates = self.n_subsets * 10
        num_positive = len(np.where(y == y_max)[0])
        num_negative = len(np.where(y == y_min)[0])
        pos_positive = np.where(y == y_max)[0]
        pos_negative = np.where(y == y_min)[0]

        if num_positive > num_negative:
            y[pos_negative] = y_max
            y[pos_positive] = y_min
            self.reverse = True

        [_, num_features] = np.shape(X)
        existing_subsets = {}
        indices_candidates_new = {}
        for i in range(0, self.n_estimators):
            self.estimators[i] = deepcopy(self.base_estimator)
            num_features_new = num_features
            mapper = np.zeros((num_features, num_features_new))
            for q in range(0, num_features_new):
                entries = random.sample(range(0, num_features), self.n_entries)
                for p in range(0, len(entries)):
                    if random.random() <= 0.5:
                        mapper[entries[p], q] = 1
                    else:
                        mapper[entries[p], q] = -1
            self.mappers[i] = deepcopy(mapper)
            X_new = np.dot(X, mapper)

            redundancy_matrix, _ = spearmanr(X_new)  # please ignore this waring for computational efficiency
            try:
                if math.isnan(redundancy_matrix):
                    redundancy_matrix = np.ones((len(X_new[0, ...]), len(X_new[0, ...])))
                    for p in range(0, len(X_new[0, ...])):
                        for q in range(p + 1, len(X_new[0, ...])):
                            redundancy_matrix[p, q] = spearmanr(X_new[..., p], X_new[..., q])[0]
                            redundancy_matrix[q, p] = redundancy_matrix[p, q]
            except TypeError:
                pass
            for p in range(0, len(X_new[0, ...])):
                for q in range(0, len(X_new[0, ...])):
                    if math.isnan(redundancy_matrix[p, q]):
                        redundancy_matrix[p, q] = 1

            redundancy = np.mean(np.mean(np.abs(redundancy_matrix)))

            infs = InfFS()
            [rank, _] = infs.infFS(X_new, y, alpha=0.75, supervision=True, verbose=False)
            try:
                self.selector[i] = deepcopy(rank[0: int(len(rank) * (1 - redundancy / 2) + 0.5)])
            except ValueError:
                pass
            X_new = X_new[..., self.selector[i]]

            X_positive = X_new[np.where(y == y_max)[0]]
            X_negative = X_new[np.where(y == y_min)[0]]

            indices_candidates = {}
            for key in indices_candidates_new:
                indices_candidates[key] = deepcopy(indices_candidates_new[key])
            for j in range(len(indices_candidates_new), n_candidates):
                indices_candidates[j] = random.sample(range(0, num_negative), num_positive)

            X_temp = np.zeros((n_candidates, len(X_positive[0]) * 2))
            for j in range(0, n_candidates):
                batch = deepcopy(X_negative[indices_candidates[j]])
                mean = np.mean(batch, axis=0)
                std = np.std(batch, ddof=1, axis=0)
                X_temp[j][0: len(mean)] = deepcopy(mean)
                X_temp[j][len(mean):] = deepcopy(std)

            mean = np.mean(X_negative, axis=0)
            std = np.std(X_negative, ddof=1, axis=0)

            weight_novelty = np.zeros((n_candidates, n_candidates))
            if len(existing_subsets) > 0:
                novelty = np.zeros(n_candidates)
                for p in range(0, n_candidates):
                    for k in range(0, len(existing_subsets)):
                        novelty[p] += 1 - len(set(indices_candidates[p]) & set(existing_subsets[k])) / \
                                      len(set(indices_candidates[p]) | set(existing_subsets[k]))
                if np.min(novelty) != np.max(novelty):
                    novelty -= np.min(novelty)
                    novelty /= np.max(novelty)
                else:
                    novelty /= np.max(novelty)
                for p in range(0, n_candidates):
                    for q in range(0, n_candidates):
                        weight_novelty[p, q] = max(novelty[p], novelty[q])

                if np.min(weight_novelty) != np.max(weight_novelty):
                    weight_novelty -= np.min(weight_novelty)
                    weight_novelty /= np.max(weight_novelty)
                else:
                    weight_novelty /= np.max(weight_novelty)

            weight_diversity = np.zeros((n_candidates, n_candidates))
            for p in range(0, n_candidates):
                for q in range(0, n_candidates):
                    weight_diversity[p, q] = 1 - len(set(indices_candidates[p]) & set(
                        indices_candidates[q])) / len(set(indices_candidates[p]) | set(indices_candidates[q]))
            weight_diversity -= np.min(weight_diversity)
            weight_diversity /= np.max(weight_diversity)

            weight_similarity = np.zeros((n_candidates, n_candidates))

            similarity = np.zeros(n_candidates)
            for p in range(0, n_candidates):
                mean_p = np.mean(X_negative[indices_candidates[p]], axis=0)
                std_p = np.std(X_negative[indices_candidates[p]], ddof=1, axis=0)
                similarity[p] = 1 - (np.sum(np.abs(std - std_p)) + np.sum(np.abs(mean - mean_p)))
            similarity -= np.min(similarity)
            similarity /= np.max(similarity)
            for p in range(0, n_candidates):
                for q in range(0, n_candidates):
                    weight_similarity[p, q] = max(similarity[p], similarity[q])
            weight_similarity -= np.min(weight_similarity)
            weight_similarity /= np.max(weight_similarity)

            alpha = (i + 1) / (self.n_estimators + 1)
            matrix = alpha * (weight_novelty + weight_diversity) / 2 + (1 - alpha) * weight_similarity
            matrix -= np.min(matrix)
            matrix /= np.max(matrix)

            infs = InfFS()
            [rank, _] = infs.infFS(matrix, None, alpha=0.75, supervision=False, verbose=False, novel=True)
            self.estimators[i] = VotingEstimator(estimator_bundle={})
            for j in range(0, self.n_subsets):
                self.estimators[i].estimator_bundle[j] = deepcopy(self.base_estimator)
                X_negative_infs = deepcopy(X_negative[indices_candidates[rank[j]]])
                X_sparse = np.concatenate((X_positive, X_negative_infs))
                y_sparse = np.concatenate((np.ones(num_positive, dtype=int), np.zeros(num_positive, dtype=int)))
                self.estimators[i].estimator_bundle[j].fit(X_sparse, y_sparse)
                existing_subsets[len(existing_subsets)] = deepcopy(indices_candidates[rank[j]])

            for j in range(self.n_subsets, self.n_subsets * 3):
                indices_candidates_new[j - self.n_subsets] = deepcopy(indices_candidates[rank[j]])

    def predict(self, X):
        label_pred_proba = self.predict_proba(X)
        label_pred = np.zeros(len(label_pred_proba), dtype=int)
        positive_pred = np.where(label_pred_proba[..., 1] > 0.5)
        label_pred[positive_pred] = 1

        return label_pred

    def predict_proba(self, X):
        label_pred_proba = np.zeros((len(X), 2))
        for i in range(0, self.n_estimators):
            X_new = np.dot(X, self.mappers[i])
            X_new = X_new[..., self.selector[i]]
            label_pred_proba += self.estimators[i].predict_proba(X_new) / self.n_estimators

        if self.reverse:
            label_pred_proba[..., 0], label_pred_proba[..., 1] = label_pred_proba[..., 1], label_pred_proba[..., 0]

        return label_pred_proba


class VotingEstimator(object):
    def __init__(self, estimator_bundle):
        self.estimator_bundle = deepcopy(estimator_bundle)
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = deepcopy(X_train)
        self.y_train = deepcopy(y_train)

    def predict_proba(self, X_test):
        label_pred_proba = np.zeros((len(X_test), 2))
        n_estimators = len(self.estimator_bundle)
        for i in range(0, n_estimators):
            label_pred_proba += self.estimator_bundle[i].predict_proba(X_test) / n_estimators

        return label_pred_proba

    def predict(self, X_test):
        label_pred_proba = self.predict_proba(X_test)
        label_pred = np.zeros(len(label_pred_proba), dtype=int)
        positive_pred = np.where(label_pred_proba[..., 1] > 0.5)
        label_pred[positive_pred] = 1

        return label_pred
