#coding=utf-8
import numpy as np
import pandas as pd
from collections import defaultdict
from Orange.classification import svm
from Orange.data import Table, Domain
from Orange.feature import Discrete, Continuous

# Domain for each column
def series2descriptor(d):
    if d.dtype is np.dtype("float") or d.dtype is np.dtype("int"):
        return Continuous(str(d.name))
    else:
        t = d.unique()
        t.sort()
        return Discrete(str(d.name), values=list(t.astype("str")))

# Create Domain
def df2domain(df):
    featurelist = [series2descriptor(df.iloc[:,col]) for col in range(len(df.columns))]
    return Domain(featurelist)

# Transform pandas.DataFrame to Orange.Table
def df2tb(data_X, data_y):
    data = data_X
    data['y'] = data_y.iloc[:, 0]
    domain = df2domain(data)
    return Table(domain, data.as_matrix())

# SVM REF class
class svmref(object):
    def __init__(self, data_X, data_y):
        """
        data_X: DataFrame object.
        data_y: DataFrame object.
        """    
        self.data_X = data_X
        self.data_y = data_y
        self.features = list(data_X.columns)
        self.n_features = len(self.features)
        self.ranking = []
        self.logging = []

    def run(self, iterations=1):
        """
        iterations: Number of iterations.
        """
        for i in range(iterations):
            ranking_per_iter = self.features_ranking_svm(name="Iterations "+str(i), random_state=64+i)
            self.ranking.append(ranking_per_iter)

    def features_ranking_svm(self, name="ranking", random_state=64):
        """Get ranking of features by REF
        random_state: Set seed in svm.SVMLearnerEasy, but it does not work.
        """
        ranking_list = []
        f_list = [col for col in self.features]
        logging_per_run = list()
        for i in range(self.n_features):
            train_X = self.data_X[f_list]
            train_y = self.data_y
            if len(train_X.columns) == 0:
                break
            tuned_learner = svm.SVMLearnerEasy(folds=5, kernel_type=svm.kernels.Linear, svm_type=svm.SVMLearner.C_SVC, random_state=random_state)
            org_data_table = df2tb(train_X, train_y)
            weights = svm.get_linear_svm_weights(tuned_learner(org_data_table), sum=False)
            internal_scores = defaultdict(float)
            for w in weights:
                magnitude = np.sqrt(sum([w_attr ** 2 for attr, w_attr in w.items()]))
                for attr, w_attr in w.items():
                    internal_scores["%s" % attr] += (w_attr / magnitude) ** 2
            features_score = []
            for i in internal_scores:
                attr_name = i.split("Orange.feature.Continuous 'N_")[1].split("'")[0]
                features_score.append((attr_name, internal_scores[i]))
            features_score.sort(lambda a, b:cmp(a[1], b[1]))
            # Results for low-score feature
            logging_per_run.append(features_score)
            ranking_list.append(features_score[0][0])
            f_list.remove(features_score[0][0])
        self.logging.append((name, logging_per_run))
        return ranking_list

    def print_logging(self, filename="ranking.txt"):
        """Print results of SVM-REF
        ranking.txt: Give ranking of features.
        logging.txt: Give procession of REF.
        """
        with open(filename, "w") as f:
            for i in range(len(self.ranking)):
                f.write("Iterations " + str(i) + ":\n")
                n = len(self.ranking[i])
                for j in range(n):
                    f.write("\tNo.%d\t%s\n" % (j, self.ranking[i][n-1-j]))
        with open("logging.txt", "w") as f:
            for logging_per_run in self.logging:
                f.write(logging_per_run[0] + ":\n")
                for L in logging_per_run[1]:
                    f.write("\t")
                    for attr in L:
                        f.write("(%s, %f) " % (attr[0], attr[1]))
                    f.write("\n")
        