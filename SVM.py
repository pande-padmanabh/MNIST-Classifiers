'''
Support Vector Machine classifier for MNIST data. 
Best Parameters are selected using simple validation
'''

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import operator
import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

dataset_file = '/home/padmanabh/Work/notMNIST.pickle'
model_path = '/home/padmanabh/Work/KNN_model.pickle'

image_size = 28
num_classes = 10
hyper_params = []

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
	
	hyper_params.append(('ovr', 'rbf', 2.5))
	
	for param_set in hyper_params:
		accuracy = createModel(param_set, train_dataset, train_labels, test_dataset, test_labels)

			
def createModel(hyper_params, train_dataset, train_labels, test_dataset, test_labels):
	function_name, kernel_name, c_value = hyper_params
	print "Starting to build SVM model (%s) with '%s' kernel and C = %f" % (function_name, kernel_name, c_value)
	classifier = SVC(decision_function_shape=function_name, kernel=kernel_name, C=c_value)
	classifier.fit(train_dataset, train_labels)

	train_pred = classifier.predict(train_dataset)
	train_accuracy = 100 * np.mean(train_labels == train_pred)
	
	test_pred = classifier.predict(test_dataset)
	test_accuracy = 100 * np.mean(test_labels == test_pred)
	print "Train Accuracy = %f, Test Accuracy = %f" % (train_accuracy, test_accuracy)
	print ""
	return test_accuracy

def reformat(dataset):	
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	return dataset
	
	
if __name__ == "__main__":
	main()