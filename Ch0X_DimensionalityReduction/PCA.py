#coding=utf-8
import numpy as np

class PCA(object):
    def __init__(self, n_component=2):
        self.n_component = n_component

    def fit(self, data_matrix):
        """
        data_matrix: 2D np.array with N rows and M columns, M is the number of features.
        """
        self.n_sample = data_matrix.shape[0]
        self.n_feature = data_matrix.shape[1]
        self.data = data_matrix
        # each value at column of matrix subtracts its mean
        data_matrix = data_matrix - np.mean(data_matrix, axis=0, keepdims=True)
        # Covariance matrix
        C = np.dot(np.transpose(data_matrix), data_matrix) / (self.n_sample - 1)
        # D = P * C * P^T
        L, P = np.linalg.eig(C)
        sorted_idx = np.argsort(L)[::-1]
        self._L = L[sorted_idx]
        self._P = P[:, sorted_idx]
        
    def get_components(self, c=None):
        """
        Get components from data using PCA algorithm.
        """
        C = self.n_component if c is None else c
        assert C <= self.n_feature
        for i in range(C):
            print "* The %dth component:" % (i + 1)
            for j in range(self.n_feature):
                print "\t%f * x%d %c " % (self._P[j, i], j, '.' if j+1 == self.n_feature else '+')

    def transform_data(self, c=None):
        C = self.n_component if c is None else c
        assert C <= self.n_feature
        return np.dot(self.data, self._P[:, :C])

    def get_eig():
        return self._L, self._P

def main():
    pass

if __name__ == '__main__':
	main()