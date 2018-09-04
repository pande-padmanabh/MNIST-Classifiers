'''
Logisgic Regression using liblinear solver and l1 regularization. 
'''

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import operator
import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
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

	hyper_params.append((DecisionTreeClassifier(criterion='entropy', max_depth=5, min_impurity_decrease=0.01), 50, 0.8, True))
	
	for params in hyper_params:
		createModel(params, train_dataset, train_labels, test_dataset, test_labels)

	print "Done"
		
def createModel(hyper_params, train_dataset, train_labels, test_dataset, test_labels):
	base_estimator = hyper_params[0]
	n_estimators = hyper_params[1]
	max_samples = hyper_params[2]
	oob_score = hyper_params[3]
	
	print "Starting to create Decision Tree with %s ..." % (str(hyper_params)) 
	classifier = BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators, \
					max_samples=max_samples, oob_score=oob_score)
						
	classifier.fit(train_dataset, train_labels)
	
	train_pred = classifier.predict(train_dataset)
	train_accuracy = 100 * np.mean(train_labels == train_pred)
	
	test_pred = classifier.predict(test_dataset)
	test_accuracy = 100 * np.mean(test_labels == test_pred)
	print "Train Accuracy = %f, Test Accuracy = %f" % (train_accuracy, test_accuracy)
	print
	
	return
	print "Confusion Matrix : "
	print confusion_matrix(test_labels, test_pred)
	print ""
	print "Report : "
	print classification_report(test_labels, test_pred)
	print "_" * 100
	
def reformat(dataset):	
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	return dataset
	
if __name__ == "__main__":
	main()