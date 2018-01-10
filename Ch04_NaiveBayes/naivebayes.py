#coding=utf-8
from __future__ import print_function

import numpy as np
from numpy import random
import logging
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)

def getFakeDataset():
    posts = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'],
    ]
    classes = [0, 1, 0, 1, 0, 1]  # 1表示为侮辱性句子, 0为普通句子
    return posts, classes

def getVocabulary(dataset):
    """get all vocabulary from dataset

    """
    vocabulary = set([])
    for document in dataset:
        vocabulary |= set(document)
    return list(vocabulary)

def getSetOfWords2Vec(vocabulary, inputSet):
    """get set of words2Vec
    """
    appearVec = [0] * len(vocabulary)
    for word in inputSet:
        if word in vocabulary:
            appearVec[vocabulary.index(word)] = 1
    return appearVec

def getBagOfWords2Vec(vocabulary, inputSet):
    """get Bag of words2Vec
    """
    appearCntVec = [0] * len(vocabulary)
    for word in inputSet:
        if word in vocabulary:
            appearCntVec[vocabulary.index(word)] += 1
    return appearCntVec

class NaiveBayesModel(object):
    """ Naive Bayesian Model"""

    def __init__(self, matrix, categories):
        self.trainMatrix = np.array(matrix)
        self.trainCategory = np.array(categories)

        # m for len of dataset
        # n for len of words2Vec
        m, n = self.trainMatrix.shape
        self.pClass1 = 1.0*sum(self.trainCategory) / m
        # Laplace calibration
        # initialize with 1
        wordsCountVector = {
            'class0': np.ones(n),
            'class1': np.ones(n),
        }
        for rowno in range(m):
            if self.trainCategory[rowno] == 0:
                wordsCountVector['class0'] += self.trainMatrix[rowno]
            else:
                wordsCountVector['class1'] += self.trainMatrix[rowno]
        # log applied to probability
        self.pWordsVector = {
            'class0': np.log(
                wordsCountVector['class0']
                / (1 + wordsCountVector['class0'].sum())
            ),
            'class1': np.log(
                wordsCountVector['class1']
                / (1 + wordsCountVector['class1'].sum())
            ),
        }

    def predict(self, inputVector):
        inputVector = np.array(inputVector)
        p0 = (
            sum(inputVector * self.pWordsVector['class0'])
            + np.log(1.0 - self.pClass1)
        )
        p1 = (
            sum(inputVector * self.pWordsVector['class1'])
            + np.log(self.pClass1)
        )
        # print('{:.3f}/{:.3f} for Class {}/{}'.format(
        #     np.exp(p0)*100, np.exp(p1)*100, 0, 1
        # ))
        if p1 > p0:
            return 1
        else:
            return 0

def testingNaiveBayes():
    postsToken, postsClass = getFakeDataset()
    vocabulary = getVocabulary(postsToken)
    trainMatrix = [
        getSetOfWords2Vec(vocabulary, post) for post in postsToken
        ]
    model = NaiveBayesModel(trainMatrix, postsClass)

    testEntry = ['love', 'my', 'dalmation']
    testPost = getSetOfWords2Vec(vocabulary, testEntry)
    print(testEntry, 'classified as: ', model.predict(testPost))

    testEntry = ['stupid', 'garbage']
    testPost = getSetOfWords2Vec(vocabulary, testEntry)
    print(testEntry, 'classified as: ', model.predict(testPost))

""" Email Classification by NaiveBayesModel"""


def getContentTokens(content):
    """ document token """
    import re
    tokens = re.split(r'\W*', content)
    return [token.lower() for token in tokens if len(token) > 2]


def testNaiveBayesToSpamEmail():
    """ test Model"""
    emails = []
    emails_class = []

    for i in range(1, 26):
        # class 1
        words = getContentTokens(open('email/spam/%d.txt' % i).read())
        emails.append(words)
        emails_class.append(1)
        # class 0
        words = getContentTokens(open('email/ham/%d.txt' % i).read())
        emails.append(words)
        emails_class.append(0)

    # generate train/test set
    random_order = random.permutation(50)
    testIndexs, trainIndexs = random_order[:10], random_order[10:]

    # generate vocabulary
    vocabulary = getVocabulary(emails)
    # train Naive Bayesian Classifier
    trainMatrix = []
    trainCategories = []
    for docIndex in trainIndexs:
        trainMatrix.append(
            getBagOfWords2Vec(vocabulary, emails[docIndex])  # Bag of words2Vec
        )
        trainCategories.append(emails_class[docIndex])
    logging.info('Train dataset is ready.')
    model = NaiveBayesModel(trainMatrix, trainCategories)
    logging.info('NaiveBayes model is trained.')

    # test set
    errorCount = 0
    for docIndex in testIndexs:
        wordVector = getBagOfWords2Vec(vocabulary, emails[docIndex])
        result = model.predict(wordVector)
        if result != emails_class[docIndex]:
            errorCount += 1
            logging.warning('classification error. Predict/Actual: {}/{}\n{}'.format(
                result,
                emails_class[docIndex],
                ' '.join(emails[docIndex])
            ))
    logging.info('the error rate is: {:.2%}'.format(1.0*errorCount/len(testIndexs)))

"""  """


def calcMostFreq(vocabulary, fullText, topN):
    import operator
    wordFrequence = {}
    for word in vocabulary:
        wordFrequence[word] = fullText.count(word)
    sortedFrequence = sorted(
        wordFrequence.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return sortedFrequence[:topN]


def getLocalWords(feed1, feed0):
    summaries = []
    summaries_class = []
    fullText = []
    minLen = min(
        len(feed1['entries']),
        len(feed0['entries'])
    )
    for i in range(minLen):
        # 第一个feed, 例子中为New York
        wordList = getContentTokens(feed1['entries'][i]['summary'])
        summaries.append(wordList)
        fullText.extend(wordList)
        summaries_class.append(1)
        # 第二个feed
        wordList = getContentTokens(feed0['entries'][i]['summary'])
        summaries.append(wordList)
        fullText.extend(wordList)
        summaries_class.append(0)
    vocabulary = getVocabulary(summaries)

    # `停用词表` -- 语言中作为冗余/结构辅助性内容的词语表
    # 多语言停用词表例子 www.ranks.nl/resources/stopwords.html
    # 去除出现次数最多的N个词
    topN = 30
    topNWords = calcMostFreq(vocabulary, fullText, topN)
    for word, _count in topNWords:
        if word in vocabulary:
            vocabulary.remove(word)

    # 生成测试集, 训练集
    random_order = random.permutation(2*minLen)
    testIndexs, trainIndexs = random_order[:20], random_order[20:]

    # 训练朴素贝叶斯分类器
    trainMatrix = []
    trainCategories = []
    for docIndex in trainIndexs:
        trainMatrix.append(getBagOfWords2Vec(vocabulary, summaries[docIndex]))
        trainCategories.append(summaries_class[docIndex])
    model = NaiveBayesModel(trainMatrix, trainCategories)

    # 进行分类测试
    errorCount = 0
    for docIndex in testIndexs:
        wordVector = getBagOfWords2Vec(vocabulary, summaries[docIndex])
        result = model.predict(wordVector)
        if result != summaries_class[docIndex]:
            errorCount += 1
            logging.warning('[classification error] Predict/Actual: {}/{}\n{}'.format(
                result,
                summaries_class[docIndex],
                ' '.join(summaries[docIndex])
            ))
    logging.info('[error rate] {:.2%}'.format(1.0*errorCount/len(testIndexs)))
    return vocabulary, model.pWordsVector


def getTopWords(ny, sf):
    vocabulary, pWordsVector = getLocalWords(ny, sf)
    top = {'NY': [], 'SF': [], }

    THRESHOLD = -6.0
    for i in range(len(pWordsVector['class0'])):
        if pWordsVector['class0'][i] > THRESHOLD:
            top['NY'].append((vocabulary[i], pWordsVector['class0'][i]))
        if pWordsVector['class1'][i] > THRESHOLD:
            top['SF'].append((vocabulary[i], pWordsVector['class1'][i]))
    import pprint
    sortedWords = {
        'SF': list(map(
            lambda x: x[0],
            sorted(top['SF'], key=lambda pair: pair[1], reverse=True)
        )),
        'NY': list(map(
            lambda x: x[0],
            sorted(top['NY'], key=lambda pair: pair[1], reverse=True)
        )),
    }
    print('=====>>  SF  <<=====')
    pprint.pprint(sortedWords['SF'])
    print('=====>>  NY  <<=====')
    pprint.pprint(sortedWords['NY'])


if __name__ == '__main__':
    testingNaiveBayes()
    testNaiveBayesToSpamEmail()

    import feedparser
    ny = feedparser.parse('http://newyork.craigslist.org/search/stp?format=rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/search/stp?format=rss')
    getTopWords(ny, sf)