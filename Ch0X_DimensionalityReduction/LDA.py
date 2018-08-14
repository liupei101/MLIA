#coding=utf-8
import numpy as np
import pandas as pd

# LDA as binary classifier
class LDA(object):
    def __init__(self, n_component=1):
        self.n_component = n_component

    def fit(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.n_sample = len(data_x.index)
        self.n_feature = len(data_x.columns)
        # prepare data as shape (n_sample, n_feature)
        data0 = data_x[data_y['y'] == 0].values
        data1 = data_x[data_y['y'] == 1].values
        # zero-mean
        u  = np.mean(data_x.values, axis=0, keepdims=True)
        u0 = np.mean(data0, axis=0, keepdims=True)
        u1 = np.mean(data1, axis=0, keepdims=True)
        # calculate sw
        sw = np.dot(np.transpose(data0 - u0), data0 - u0) + np.dot(np.transpose(data1 - u1), data1 - u1)
        # calculate sb
        sb = data0.shape[0] * np.dot(np.transpose(u0 - u), u0 - u) + data1.shape[0] * np.dot(np.transpose(u1 - u), u1 - u)
        # sw^-1 * sb
        C = np.linalg.inv(sw) * sb
        # D = P * C * P^T
        L, P = np.linalg.eig(C)
        sorted_idx = np.argsort(L)[::-1]
        self._L = L[sorted_idx]
        self._P = P[:, sorted_idx]
        
    def get_components(self, c=None):
        """
        Get components from data using LDA algorithm.
        Note: the dimension of transformed_data is equal or less than K - 1.(K is the number of different y)
        """
        C = self.n_component if c is None else c
        assert C <= 1
        for i in range(C):
            print "* The %dth component:" % (i + 1)
            for j in range(self.n_feature):
                print "\t%f * x%d %c " % (self._P[j, i], j, '.' if j+1 == self.n_feature else '+')

    def get_transformed_data(self, c=None):
        C = self.n_component if c is None else c
        assert C <= 1
        transformed_data = np.dot(self.data_x.values, self._P[:, :C])
        return pd.DataFrame(transformed_data, columns=self.data_x.columns), self.data_y

    def get_eig():
        return self._L, self._P

def main():
    pass

if __name__ == '__main__':
	main()