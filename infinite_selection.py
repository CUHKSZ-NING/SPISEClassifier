import math
import numpy as np
from scipy import stats
from copy import deepcopy


class InfFS:
    def __init__(self):
        self.stop_warning = True

    # Take in input the matrix e the label vector and return a matrix
    # of data for every different label.
    def takeLabel(self, X_train, y_train):
        self.stop_warning = True
        counter = X_train.shape[0] - 1
        s_n = X_train
        s_p = X_train
        while 1:
            if y_train[counter] == 1:
                s_n = np.delete(s_n, counter, axis=0)
            else:
                s_p = np.delete(s_p, counter, axis=0)
            counter = counter - 1
            if counter == -1:
                break
        return s_p, s_n

    # Function that help to define priors_corr.
    def defPriorsCorr(self, mu_s_n, mu_s_p):
        self.stop_warning = True
        pcorr = mu_s_p
        counter = 0
        while counter < len(pcorr):
            pcorr[counter] = (pcorr[counter] - mu_s_n[counter]) * (
                pcorr[counter] - mu_s_n[counter]
            )
            counter = counter + 1
        return pcorr

    # Function to subtract the min value of the matrix to all it's elements.
    def SubtractMin(self, corr_ij):
        self.stop_warning = True
        m = 10100
        for i in range(0, corr_ij.shape[0]):  # Find the min.
            for j in range(0, corr_ij.shape[1]):
                if corr_ij[i, j] < m:
                    m = corr_ij[i, j]

        for i in range(0, corr_ij.shape[0]):  # Subtract the min value.
            for j in range(0, corr_ij.shape[1]):
                corr_ij[i, j] = corr_ij[i, j] - m

        return corr_ij

    # Function to divide every element of the matrix to his maximum value.
    def DivideByMax(self, corr_ij):
        self.stop_warning = True
        m = -1
        for i in range(0, corr_ij.shape[0]):  # Find the max.s
            for j in range(0, corr_ij.shape[1]):
                if corr_ij[i, j] > m:
                    m = corr_ij[i, j]

        for i in range(0, corr_ij.shape[0]):  # Divide by the maximum value.
            for j in range(0, corr_ij.shape[1]):
                corr_ij[i, j] = corr_ij[i, j] / m

        return corr_ij

    # Handmaded bsxfunction that take the max.
    def bsxfun(self, STD):
        self.stop_warning = True
        m = np.zeros((STD.shape[0], STD.shape[0]))
        for i in range(0, STD.shape[0]):
            for j in range(0, STD.shape[0]):
                if STD[i] > STD[j]:
                    m[i, j] = STD[i]
                else:
                    m[i, j] = STD[j]
        return m

    def infFS(self, X_train, y_train, alpha, supervision, verbose, is_examples=False, target=None, novel=False):
        # Start of point one.
        if not novel:
            if supervision:
                s_p, s_n = self.takeLabel(X_train, y_train)
                mu_s_n = s_n.mean(0)
                mu_s_p = s_p.mean(0)
                priors_corr = self.defPriorsCorr(mu_s_n, mu_s_p)
                st = np.power(np.std(s_p, ddof=1, axis=0), 2)
                st = st + np.power(np.std(s_n, ddof=1, axis=0), 2)
                for i in range(0, len(st)):
                    if st[i] == 0:
                        st[i] = 10000
                corr_ij = priors_corr
                for i in range(0, len(corr_ij)):
                    corr_ij[i] = corr_ij[i] / st[i]
                corr_ij = np.dot(corr_ij.T[:, None], corr_ij[None, :])
                corr_ij = self.SubtractMin(corr_ij)
                corr_ij = self.DivideByMax(corr_ij)
            else:
                if not is_examples:
                    corr_ij, pval = stats.spearmanr(X_train)
                else:
                    corr_ij = get_euclidean(X_train)
                try:
                    for i in range(0, corr_ij.shape[0]):
                        for j in range(0, corr_ij.shape[1]):
                            if (
                                math.isnan(corr_ij[i, j])
                                or corr_ij[i, j] < -1
                                or corr_ij[i, j] > 1
                            ):
                                corr_ij[i, j] = 0
                except AttributeError:
                    pass
    
            # After if.
            if not is_examples:
                STD = np.std(X_train, ddof=1, axis=0)
                STDMatrix = self.bsxfun(STD)
            else:
                STDMatrix = get_manhattan(X_train, target)
            STDMatrix = self.SubtractMin(STDMatrix)
            sigma_ij = self.DivideByMax(STDMatrix)
            for i in range(0, sigma_ij.shape[0]):
                for j in range(0, sigma_ij.shape[1]):
                    if (
                        math.isnan(sigma_ij[i, j])
                        or sigma_ij[i, j] < -1
                        or sigma_ij[i, j] > 1
                    ):
                        sigma_ij[i, j] = 0
    
            # End of point one.
    
            # Start of the point two.
            if verbose:
                print("2) Building the graph G = <V,E> \n")
            A = alpha * corr_ij + (1 - alpha) * sigma_ij
        # End of the point two.
        else:
            A = deepcopy(X_train)

        # Start of the point three.
        if verbose:
            print("3) Letting paths tend to infinite \n")

        I_matrix = np.identity(A.shape[0])
        try:
            r = 0.9 / max(np.linalg.eigvals(A))  # Setting the r values.
        except np.linalg.LinAlgError:
            pass
        y = I_matrix - (r * A)
        S = np.linalg.inv(y) - I_matrix
        # End of point three.

        # Start of point four.
        if verbose:
            print("4) Estimating energy scores \n")

        WEIGHT = np.sum(S, axis=1)
        # End of point four.

        # Start of point five.
        if verbose:
            print("5) Features ranking")

        RANKED = np.argsort(WEIGHT)
        RANKED = np.flip(RANKED, 0)
        RANKED = RANKED.T
        WEIGHT = WEIGHT.T
        return RANKED, WEIGHT
        # End of point five.


def get_euclidean(X_train):
    num_features = len(X_train[0])
    corr_ij = np.zeros((num_features, num_features))
    for i in range(0, num_features):
        corr_ij[i, i] = 0
        for j in range(i + 1, num_features):
            corr_ij[i, j] = np.linalg.norm(X_train[..., i] - X_train[..., j])
            corr_ij[j, i] = corr_ij[i, j]
    
    return corr_ij


def get_manhattan(X_train, target):
    num_features = len(X_train[0])
    sigma_ij = np.zeros((num_features, num_features))
    manhattan_distance = np.zeros(num_features)
    for i in range(0, num_features):
        manhattan_distance[i] = sum(map(abs, X_train[..., i] - target))
    
    manhattan_distance -= min(manhattan_distance)

    for i in range(0, num_features):
        for j in range(0, num_features):
            sigma_ij[i, j] = max(manhattan_distance[i], manhattan_distance[j])
    
    return sigma_ij
    