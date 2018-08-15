import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import os
import sys
import glob
import matplotlib
import numpy as np
from time import sleep
from scipy import ndimage
import cPickle as pickle
from traceback import print_exc

train_folder_name = '/home/padmanabh/Work/notMNIST_large'
test_folder_name = '/home/padmanabh/Work/notMNIST_small'
dataset_file = '/home/padmanabh/Work/notMNIST.pickle'

train_size = 40000
test_size = 5000
validation_size = 5000
	
num_classes = 10
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def main():
	np.random.seed(133)
	create_pickles('train')
	create_pickles('test')

	print "Starting to create dataset from pickle files"
	print ""
	generate_dataset(train_size, test_size, validation_size)
	print ""
	print "Exiting with status 0"
	
def generate_dataset(train_size, test_size, validation_size=0):
	model_dataset = dict()
	if os.path.exists(dataset_file):
		os.remove(dataset_file)
	
	print "Expected testing dataset size = %d" % (test_size)
	test_keys, test_labels, validation_keys, validation_labels = \
		append_dataset(test_folder_name, test_size)

	print "Expected training dataset size = %d" % (train_size)
	print "Expected validation dataset size = %d" % (validation_size)
	train_keys, train_labels, validation_keys, validation_labels = \
		append_dataset(train_folder_name, train_size, validation_size)

	model_dataset['test_keys'] = test_keys
	model_dataset['test_labels'] = test_labels
	model_dataset['train_keys'] = train_keys
	model_dataset['train_labels'] = train_labels

	if validation_size:
		model_dataset['validation_keys'] = validation_keys
		model_dataset['validation_labels'] = validation_labels

	print "Dumping dataset to file '%s'" % (dataset_file)
	try:
		handle = open(dataset_file, 'wb')
		pickle.dump(model_dataset, handle, pickle.HIGHEST_PROTOCOL)
		handle.close()
		print "Dataset pickle file successfully created"
	except Exception as  e:
		print "Unable to create dataset pickle file"
		
def create_pickles(taskname):
	print "Starting pickle creation for %s" % (taskname)
	if taskname == 'train':
		folder_name = train_folder_name
	elif taskname == 'test':
		folder_name = test_folder_name
	else:
		print "Inappropriate task provided"
		return
		
	contents = os.listdir(folder_name)
	class_list = [elem for elem in contents if '.pickle' not in elem]
	count = 0
	print "Starting to create pickle files of individual classes"
	print ""
	for class_name in class_list:
		pickle_file = class_name + '.pickle'
		if pickle_file in contents:
			print "Pickle file '%s' already exists" % (pickle_file)
			sleep(1)
			continue

		print "Creating pickle file '%s'" % (pickle_file)
		class_data = load_dataset(folder_name + '/' + class_name)

		try:
			with open(folder_name + '/' + pickle_file, 'wb') as handle:
				pickle.dump(class_data, handle, pickle.HIGHEST_PROTOCOL)
				print "Pickle file (%s) created for class '%s'" % (pickle_file, class_name)
				print ""
				count += 1
		except Exception as e:
			print "Unable to create pickle file (%s) for class '%s'" % (pickle_file, class_name)
			
	print "%d pickle files for %s are successfully created" % (count, taskname)	
	print ""
	
def load_dataset(folder_name):
	print "Checking into folder : ", folder_name
	image_files = os.listdir(folder_name)
	sleep(5)
	dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
	print "Loading data ..."
	idx = 0
	
	for image in image_files:
		image_file = folder_name + '/' + image
		try:
			image_data = (ndimage.imread(image_file).astype(float) - 0.5 * pixel_depth) / pixel_depth
			dataset[idx, :, :] = image_data
			idx += 1
		except Exception as e:
			print "Could not read the image file '%s'" % (image_file)

	print "Dumping %d datapoints to the pickle file" % (idx)
	return dataset[:idx, :, :]
	
def append_dataset(folder_name, dataset_size, validation_size=0):
	print "Starting to generate dataset from '%s'" %  (folder_name)
	print ""
	pickle_files = [elem for elem in os.listdir(folder_name) if '.pickle' in elem]
	num_classes = len(pickle_files)
	
	dataset_keys = np.ndarray(shape=(dataset_size, image_size, image_size), dtype=np.float32)
	dataset_labels = np.ndarray(shape=(dataset_size), dtype=np.int32)

	validation_keys = np.ndarray(shape=(validation_size, image_size, image_size), dtype=np.float32)
	validation_labels = np.ndarray(shape=(validation_size), dtype=np.int32)

	start_idx = 0
	end_idx = 0
	start_v_idx = 0
	end_v_idx = 0
	partition_size = dataset_size / num_classes
	validation_partition_size = validation_size / num_classes
		
	for label, class_pickle in enumerate(pickle_files):
		start_idx = end_idx
		end_idx = start_idx + partition_size

		start_v_idx = end_v_idx
		end_v_idx = start_v_idx + validation_partition_size

		print "Adding data from file '%s' for class '%d'" % (class_pickle, label)
		with open(os.path.join(folder_name, class_pickle), 'rb') as handle:
			class_data = pickle.load(handle)
			dataset_keys[start_idx:end_idx, :, :] = class_data[0:partition_size, :, :]
			dataset_labels[start_idx:end_idx] = label

			if validation_size:
				validation_keys[start_v_idx:end_v_idx, :, :] = \
					class_data[partition_size:partition_size + validation_partition_size, :, :]
				validation_labels[start_v_idx:end_v_idx] = label

			handle.close()
			
		print "Data added for class '%d'" % (label)
		print ""
				
	dataset_keys, dataset_labels = shuffle_dataset(dataset_keys, dataset_labels)	
	print "Dataset created with size = %d" % (len(dataset_labels)) 
	if validation_size:
		validation_keys, validation_labels = shuffle_dataset(validation_keys, validation_labels)	
		print "Vallidation dataset created with size = %d" % (len(validation_labels)) 
	print ""

	return dataset_keys, dataset_labels, validation_keys, validation_labels
		
def shuffle_dataset(keys, labels):
	permutation = np.random.permutation(len(labels))
	keys = keys[permutation, :, :]
	labels = labels[permutation]
	return keys, labels
		
if __name__ == "__main__":
	main()
