#coding=utf-8

from __future__ import print_function

import math
import operator
import pickle
from collections import defaultdict

import numpy as np

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)

class Dataset(object):
    def __init__(self, rawDataset):
        # rawDataset: np.array
        self.rawDataset = np.array(rawDataset)

    @property
    def shape(self):
        return self.rawDataset.shape

    @property
    def classList(self):
        return self.rawDataset[:, -1].tolist()

    @property
    def shannonEntropy(self):
        # Count for every class label
        labelCounts = defaultdict(int)
        for featVec in self.rawDataset:
            label = featVec[-1]
            labelCounts[label] += 1
        # Entropy for label
        # H(p) = - sigma(p_i * log(2, p_i))
        entropy = 0.0
        numEntries = self.shape[0]
        for label in labelCounts:
            probability = 1.0 * labelCounts[label] / numEntries
            entropy -= probability * math.log(probability, 2)
        return entropy

    def split(self, axis):
        # remove axis from dataset
        subDatasets = defaultdict(list)
        for featureVector in self.rawDataset:
            value = featureVector[axis]
            subFeatureVector = (
                featureVector[:axis].tolist()
                + featureVector[axis+1:].tolist()
            )
            subDatasets[value].append(subFeatureVector)
        return (
            list(subDatasets.keys()),
            list(map(self.__class__, subDatasets.values())),
        )

    def ChooseBestSplitFeature(self):
        m, n = self.shape
        numFeatures = n - 1
        baseEntropy = self.shannonEntropy
        # record best
        best = 
        {
            'gain': 0.0,
            'feature': -1
        }
        # loop for every featrue
        for featureIndex in range(numFeatures):
            _labels, subDatasets = self.split(featureIndex)
            # Get entropy after split
            newEntropy = 0.0
            for subDataset in subDatasets:
                sub_m, _sub_n = subDataset.shape
                probability = 1.0 * sub_m / m
                newEntropy += probability * subDataset.shannonEntropy
            # the smaller newEntropy values, the better tree is
            infoGain = baseEntropy - newEntropy
            if infoGain > best['gain']:
                best['gain'] = infoGain
                best['feature'] = featureIndex
        return best['feature']

def GetFakeDataset():
    dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return Dataset(dataset), labels

class DecisionTree(object):
    def __init__(self, dataset, labels):
        if dataset and labels:
            self.labels = labels
            self.tree = self.BuildTree(Dataset(dataset), self.labels)

    def SaveToFile(self, filename):
        with open(filename, 'w') as outfile:
            pickle.dump(self, outfile)

    @staticmethod
    def LoadFromFile(filename):
        with open(filename, 'r') as infile:
            tree = pickle.load(infile)
        return tree

    @staticmethod
    def GetMajorityClass(classList):
        classCount = defaultdict(int)
        for vote in classList:
            classCount[vote] += 1
        sortedClassCount = sorted(
            classCount.items(),
            key=operator.itemgetter(1),
            reverse=True
        )
        return sortedClassCount[0][0]

    def BuildTree(self, dataset, labels):
        labels = labels[:]

        classList = dataset.classList
        # when elements of classList is same
        # return a element of classList
        if classList.count(classList[0]) == len(classList): 
            return classList[0]

        # when running out all features
        # return the class which occured most 
        _m, n = dataset.shape
        if n == 1:
            return self.GetMajorityClass(classList)

        # select best feature to split dataset
        bestFeatureIndex = dataset.ChooseBestSplitFeature()
        bestFeatureLabel = labels[bestFeatureIndex]
        del(labels[bestFeatureIndex])
        logging.info('Spliting by Feature {0}({1})'.format(
            bestFeatureLabel,
            bestFeatureIndex
        ))

        decisionTree = {
            bestFeatureLabel: {},
        }

        # build tree for every subDataset
        subLabels, subDatasets = dataset.split(bestFeatureIndex)
        logging.info('labels:{0} for Feature {1}'.format(subLabels, bestFeatureLabel))
        for subLabel, subDataset in zip(subLabels, subDatasets):
            logging.info('Building subtree of value `{0}`'.format(subLabel))
            decisionTree[bestFeatureLabel][subLabel] = self.BuildTree(
                subDataset,
                labels
            )
            logging.info('Subtree `{0}` built'.format(subLabel))
        return decisionTree

    def predict(self, inputVector):
        return self.GetClassOfVector(self.tree, self.labels, inputVector)

    def GetClassOfVector(self, decisionTree, featureLabels, inputVector):
        featureLabel = decisionTree.keys()[0]
        subDecisionTree = decisionTree[featureLabel]
        featureIndex = featureLabels.index(featureLabel)

        downKey = inputVector[featureIndex]
        downNode = subDecisionTree[downKey]

        if isinstance(downNode, dict):
            classLabel = self.GetClassOfVector(
                downNode, featureLabels,
                inputVector
            )
        else:
            classLabel = downNode
        return classLabel

    @property
    def depth(self):
        return self.GetTreeDepth(self.tree)

    @classmethod
    def GetTreeDepth(cls, tree):
        max_depth = 0
        featureLabel = tree.keys()[0]
        subDecisionTree = tree[featureLabel]
        for featureValue in subDecisionTree:
            if isinstance(subDecisionTree[featureValue], dict):
                depth = 1 + cls.GetTreeDepth(subDecisionTree[featureValue])
            else:
                depth = 1

            max_depth = max(depth, max_depth)
        return max_depth

    @property
    def num_leaves(self):
        return self.GetNumLeaves(self.tree)

    @classmethod
    def GetNumLeaves(cls, tree):
        num = 0
        featureLabel = tree.keys()[0]
        subDecisionTree = tree[featureLabel]
        for featureValue in subDecisionTree:
            if isinstance(subDecisionTree[featureValue], dict):
                num += cls.GetNumLeaves(subDecisionTree[featureValue])
            else:
                num += 1
        return num

    @property
    def feature_label(self):
        return self.tree.keys()[0]

    def GetSubTree(self, feature_value):
        tree = self.__class__(None, None)
        tree.tree = self.tree[self.feature_label][feature_value]
        return tree

    @classmethod
    def GetRetrieveTree(cls, index):
        trees = (
            {'no surfacing': {
                0: 'no',
                1: {'flippers':
                    {0: 'no', 1: 'yes'}}
            }},
            {'no surfacing': {
                0: 'no',
                1: {'flippers':
                    {0: {'head':
                             {0: 'no', 1: 'yes'}},
                     1:'no'}
            }}},
        )
        tree = cls(None, None)
        tree.tree = trees[index]
        return tree

def LoadLensesData(filename):
    with open(filename) as infile:
        lensesDataset = []
        for line in infile:
            trainVector = line.strip().split('\t')
            lensesDataset.append(trainVector)
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate', ]
    lenseTree = DecisionTree(lensesDataset, lensesLabels)
    return lenseTree

import matplotlib.pyplot as plt

class DecisionTreePlotter(object):

    DECISION_NODE = {
        'boxstyle': 'sawtooth',
        'fc': '0.8',
    }
    LEAF_NODE = {
        'boxstyle': 'round4',
        'fc': '0.8',
    }
    ARROW_ARGS = {
        'arrowstyle': '<-',
    }

    def __init__(self, tree):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        self.ax1 = plt.subplot(111, frameon=False, xticks=[], yticks=[])
        self.width = 1.0*tree.num_leaves
        self.depth = 1.0*tree.depth
        self.offset = {
            'x': -0.5/self.width,
            'y': 1.0
        }
        self.plot_tree(tree, (0.5, 1.0), '')
        plt.show()

    def plot_mid_text(self, text, centerPoint, parentPoint):
        xMid = (parentPoint[0] - centerPoint[0]) / 2.0 + centerPoint[0]
        yMid = (parentPoint[1] - centerPoint[1]) / 2.0 + centerPoint[1]
        self.ax1.text(xMid, yMid, text)

    def plot_node(self, text, centerPoint, parentPoint, node_type):
        self.ax1.annotate(
            text,
            xy=parentPoint, xycoords='axes fraction',
            xytext=centerPoint, textcoords='axes fraction',
            va='center', ha='center',
            bbox=node_type, arrowprops=DecisionTreePlotter.ARROW_ARGS
        )

    def plot_tree(self, tree, parentPoint, text):
        num_leaves = tree.num_leaves
        featureLabel = tree.feature_label
        centerPoint = (
            self.offset['x'] + (1.0 + num_leaves) / 2.0 / self.width,
            self.offset['y']
        )
        self.plot_mid_text(text, centerPoint, parentPoint)
        self.plot_node(
            featureLabel,
            centerPoint, parentPoint,
            DecisionTreePlotter.DECISION_NODE
        )
        subDecisionTree = tree.tree[featureLabel]
        self.offset['y'] -= 1.0/self.depth
        for featureValue in subDecisionTree:
            if isinstance(subDecisionTree[featureValue], dict):
                self.plot_tree(
                    tree.GetSubTree(featureValue),
                    centerPoint,
                    str(featureValue)
                )
            else:
                self.offset['x'] += 1.0 / self.width
                self.plot_node(
                    subDecisionTree[featureValue],
                    (self.offset['x'], self.offset['y']),
                    centerPoint,
                    DecisionTreePlotter.LEAF_NODE
                )
                self.plot_mid_text(
                    str(featureValue),
                    (self.offset['x'], self.offset['y']),
                    centerPoint
                )
        self.offset['y'] += 1.0 / self.depth

if __name__ == '__main__':
    tree = LoadLensesData('lenses.txt')
    print(tree.depth)
    t = DecisionTree.GetRetrieveTree(0)
    print(t.depth, t.num_leaves)
    plotter = DecisionTreePlotter(t)