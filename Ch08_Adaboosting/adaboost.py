#coding=utf-8
"""
Adaboost -- Adaptive boosting
"""

import logging
import numpy as np

logging.basicConfig(
	level=logging.DEBUG,
	format='[%(levelname)s %(module)s line:%(lineno)d] % (message)s',
)
TRACE = logging.DEBUG - 1

def load_fake_dataset():
    dataset = np.matrix([
        [1.0, 2.1],
        [2.0, 1.1],
        [1.3, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ])
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataset, labels

def load_dataset_from_file(filename):
    dataset = []
    labels = []
    num_features = None
    with open(filename) as infile:
        for line in infile:
            line = line.strip().split('\t')
            if num_features is None:
                num_features = len(line)
            dataset.append(list(map(float, line[:-1])))
            labels.append(float(line[-1]))
        return dataset, labels

# weak classifier(base)
class DecisionStump(object):
	def __init__(self, dataset):
		self.dataset = dataset

	# a dimension of dataset is compared with threshold 
	def predict(self, dimension, threshold_val, inequal):
		m, _n = self.dataset.shape
        pred = np.ones((m, 1))
        if inequal == 'lt':
            pred[self.dataset[:, dimension] <= threshold_val] = -1.0
        elif inequal == 'gt':
            pred[self.dataset[:, dimension] > threshold_val] = -1.0
        return pred

class AdaBoostDecisionStump(object):
	def __init__(self, dataset, labels, max_iter=40):
		self.dataset = np.mat(dataset)
		self.labels = np.mat(labels).T
		self.m, self.n = self.dataset.shape
		self.train(max_iter=max_iter)

	def build_stump(self, D):
        stump = DecisionStump(self.dataset)
        num_steps = 10.0  # length of step for all thresholds
        best_stump_info = {}  # record best weak classifier under D(Probability distributions of samples)
        best_predict_values = np.mat(np.zeros((self.m, 1)))
        min_error = 0x3f3f3f3f  # init error sum, to +infinity
        # loop for every dimension
        for i in range(self.n):
            feature_min = self.dataset[:, i].min()
            feature_max = self.dataset[:, i].max()
            step = (feature_max - feature_min) / num_steps
            # loop over all range in current dimension
            for j in range(-1, int(num_steps) + 1):  
                for inequal in ['lt', 'gt']:
                    threshold_val = feature_min + float(j) * step
                    predicted_values = stump.predict(i, threshold_val, inequal)
                    # initialize errors[] to 1
                    errors = numpy.mat(numpy.ones((self.m, 1)))
                    errors[predicted_values == self.labels] = 0
                    # calculate weighted errors under D(Probability distributions of samples)
                    weighted_errors = D.T * errors
                    logging.log(TRACE, '[Split] dimension {:d}, threshold {:.2f} threshold inequal: {:s}'.format(
                        i, threshold_val, inequal
                    ))
                    logging.log(TRACE, '[Split] Weighted errors is {:.3f}'.format(weighted_errors[0, 0]))
                    # Get best weak classifier according to weighted error
                    if weighted_errors < min_error:
                        min_error = weighted_errors
                        best_predict_values = predicted_values.copy()
                        best_stump_info['dimension'] = i
                        best_stump_info['threshold'] = threshold_val
                        best_stump_info['inequal'] = inequal
        return best_stump_info, min_error, best_predict_values

    def train(self, max_iter):
    	weak_classifiers = []
    	D = np.mat(np.ones((self.m, 1)) / self.m)
    	aggregated_predict = np.mat(np.zeros((self.m, 1)))
        for i in range(max_iter):
            stump_info, error, predict = self.build_stump(D)
            logging.debug('D: {}'.format(D.T))
            # $alpha = 1/2 * log(\frac{1-e_m}{e_m})$
            alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
            stump_info['alpha'] = alpha
            weak_classifiers.append(stump_info)  # store Stump Params in Array
            logging.debug('predict: {}'.format(predict.T))
            # update D(Probability distributions of samples)
            exponent = np.multiply(-1 * alpha * self.labels, predict)
            D = np.multiply(D, np.exp(exponent))
            D = D / D.sum()
            # Get final classifier by add all weak classifers
            aggregated_predict += alpha * predict
            logging.debug('aggregated predict: {}'.format(aggregated_predict.T))
            aggregated_errors = np.multiply(
                np.sign(aggregated_predict) != self.labels,
                np.ones((self.m, 1))
            )
            errorRate = aggregated_errors.sum() / self.m
            logging.info('Total error: {}'.format(errorRate))
            if errorRate == 0.0:
                break
        self.classifiers = weak_classifiers
        self.aggregated_predict = aggregated_predict

    def predict(self, dataset):
        dataset = np.mat(dataset)
        stump = DecisionStump(dataset)
        m, _n = dataset.shape
        aggregated_estimate = np.mat(np.zeros((m, 1)))
        for classifier in self.classifiers:
            logging.info('Applying stumb: {}'.format(classifier))
            weight = classifier['alpha']
            pred = stump.predict(
                classifier['dimension'],
                classifier['threshold'],
                classifier['inequal']
            )
            aggregated_estimate += weight * pred
            logging.info(aggregated_estimate)
        return np.sign(aggregated_estimate)

def plotROCCurve(predStrengths, labels):
    import matplotlib.pyplot as plt
    cursor = (1.0, 1.0)
    ySum = 0.0  # variable to calculate AUC
    numPositiveClass = sum(numpy.array(labels) == 1.0)
    step = {
        'x': 1.0 / numPositiveClass,
        'y': 1.0 / (len(labels) - numPositiveClass),
    }
    sortedIndicies = predStrengths.A1.argsort()  # get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies:
        if labels[index] == 1.0:
            deltaX = 0
            deltaY = step['x']
        else:
            deltaX = step['y']
            deltaY = 0
            ySum += cursor[1]
        # draw line from cursor to (cursor[0]-deltaX, cursor[1]-deltaY)
        logging.debug('Drawing line from {} -> {}'.format(
            cursor, (cursor[0]-deltaX, cursor[1]-deltaY)
        ))
        ax.plot(
            [cursor[0], cursor[0]-deltaX],
            [cursor[1], cursor[1]-deltaY],
            c='b'
        )
        cursor = (cursor[0] - deltaX, cursor[1] - deltaY)
    ax.plot([0, 1], [0, 1], 'b--')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    logging.info('Area Under the Curve: {}'.format(ySum * step['y']))

def main():
    import pprint
    dataset, labels = load_fake_dataset()
    model = AdaBoostDecisionStump(dataset, labels)
    logging.info('Classifiers: {}'.format(pprint.pformat(model.classifiers)))
    logging.info('Result (pred/true):\n{}'.format(zip(
        model.predict(dataset).A1.tolist(),
        labels
    )))

    plotROCCurve(model.aggregated_predict, labels)
    # test for Dataset File 
    # dataset, labels = load_dataset_from_file('horseColicTraining2.txt')
    # model = AdaBoostDecisionStump(dataset, labels)
    # plotROCCurve(model.aggregated_predict, labels)

if __name__ == '__main__':
    main()