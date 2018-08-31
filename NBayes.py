'''
Naive Bayes classifier for MNIST data.
This includes Gaussian NB, Multinomial NB and Bernoulli NB.
'''

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import operator
import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

dataset_file = '/home/padmanabh/Work/notMNIST.pickle'
model_path = '/home/padmanabh/Work/KNN_model.pickle'

image_size = 28
num_classes = 10

def main():
	dataset = pickle.load(open(dataset_file, 'rb'))
	train_dataset = dataset['train_keys']
	train_labels = dataset['train_labels']
	test_dataset = dataset['test_keys']
	test_labels = dataset['test_labels']
	valid_dataset = dataset['validation_keys']
	valid_labels = dataset['validation_labels']
	
	del dataset
	train_dataset = reformat(train_dataset)
	test_dataset = reformat(test_dataset)
	valid_dataset = reformat(valid_dataset)

	print('Training set   :', train_dataset.shape, train_labels.shape)
	print('Validation set : ', valid_dataset.shape, valid_labels.shape)
	print('Test set       : ', test_dataset.shape, test_labels.shape)

	createModel('GNB', train_dataset, train_labels, test_dataset, test_labels)
	createModel('MNB', train_dataset, train_labels, test_dataset, test_labels)
	createModel('BNB', train_dataset, train_labels, test_dataset, test_labels)
	print "Done"
		
def createModel(nb_type, train_dataset, train_labels, test_dataset, test_labels):
	print "Starting to build Gaussian Naive Bayes model..."
	if nb_type == 'GNB':
		classifier = GaussianNB()
	elif nb_type == 'MNB':
		classifier = MultinomialNB()
	elif nb_type == 'BNB':
		classifier = BernoulliNB()
	else:
		return 0

	classifier.fit(train_dataset, train_labels)
	
	train_pred = classifier.predict(train_dataset)
	train_accuracy = 100 * np.mean(train_labels == train_pred)
	
	test_pred = classifier.predict(test_dataset)
	test_accuracy = 100 * np.mean(test_labels == test_pred)
	print "Train Accuracy = %f, Test Accuracy = %f" % (train_accuracy, test_accuracy)
	print "Confusion Matrix : "
	print confusion_matrix(test_labels, test_pred)
	print ""
	print "Report : "
	print classification_report(test_labels, test_pred)
	print "_" * 100
	
def reformat(dataset):	
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	dataset = np.array([row + 0.5 for row in dataset])
	return dataset
	
if __name__ == "__main__":
	main()