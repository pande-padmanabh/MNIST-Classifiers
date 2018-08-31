'''
Simple K Nearest Neighbors based model with distance priority.
It computes the best k_value using simple validation technique. 
'''

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import operator
import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

dataset_file = '/home/padmanabh/Work/notMNIST.pickle'
model_path = '/home/padmanabh/Work/KNN_model.pickle'

image_size = 28
num_classes = 10

def main():
	dataset = pickle.load(open(dataset_file, 'rb'))
	train_dataset = dataset['train_keys'][:5000]
	train_labels = dataset['train_labels'][:5000]
	test_dataset = dataset['test_keys'][:1000]
	test_labels = dataset['test_labels'][:1000]
	valid_dataset = dataset['validation_keys'][:1000]
	valid_labels = dataset['validation_labels'][:1000]
	
	del dataset
	print('Training set   :', train_dataset.shape, train_labels.shape)
	print('Validation set : ', valid_dataset.shape, valid_labels.shape)
	print('Test set       : ', test_dataset.shape, test_labels.shape)
	
	train_dataset, train_labels = reformat(train_dataset, train_labels)
	test_dataset,  _ = reformat(test_dataset, test_labels)
	valid_dataset, _ = reformat(valid_dataset, valid_labels)

	accuracy = []
	for k_value in range(1, 11):
		value = createModel(k_value, train_dataset, train_labels, valid_dataset, valid_labels, None, None)
		accuracy.append(value + 0.01 * k_value)

	print "Accuracy for K = 1 to 10 : ", accuracy
	print ""
	
	max_index, max_value = max(enumerate(accuracy), key=operator.itemgetter(1))
	print "Creating final model with K = ", max_index + 1
	createModel(max_index + 1, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)

	return

def createModel(k_value, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
	print "Starting to build KNN model with K = %d ..." % (k_value)
	classifier = KNN(n_neighbors=k_value, weights='distance', n_jobs=-1, p=2)
	classifier.fit(train_dataset, train_labels)
	
	valid_pred = np.argmax(classifier.predict(valid_dataset), 1)
	accuracy = 100 * np.mean(valid_labels == valid_pred)
	print "Accuracy with K = %02d is %.2f" % (k_value, accuracy)
	
	if test_dataset is not None:
		test_pred = np.argmax(classifier.predict(test_dataset), 1)
		accuracy = 100 * np.mean(test_labels == test_pred)
		print "Test Accuracy with K = %02d is %.2f" % (k_value, accuracy)
		print "Confusion Matrix : "
		print confusion_matrix(test_labels, test_pred)
		print "Report : "
		print classification_report(test_labels, test_pred)
		with open(model_path, 'wb') as handle:
			pickle.dump(classifier, handle)
			print "KNN Model saved at ", model_path
	
	print ""
	return accuracy

def reformat(dataset, labels):	
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	labels = (np.arange(num_classes) == labels[:, None]).astype(np.float32)
	return dataset, labels
	
	
if __name__ == "__main__":
	main()