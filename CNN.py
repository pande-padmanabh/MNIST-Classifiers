import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import tensorflow as tf
import cPickle as pickle
import time

dataset_file = '/home/padmanabh/Work/notMNIST.pickle'
model_path = '/home/padmanabh/Work/CNN_model.ckpt'

image_size = 28
num_classes = 10

# parameters for convolution
patch_size = 5
depth = 32
num_channels = 1

# learning parameters
batch_size = 1024
hidden_size = 256
num_steps = 1001
learning_rate = 0.001

# parameters for regularization
beta = 0.1

# parameters for dropout
keep_prob = 0.5

def main():
	dataset = pickle.load(open(dataset_file, 'rb'))
	train_dataset = dataset['train_keys']
	train_labels = dataset['train_labels']
	test_dataset = dataset['test_keys']
	test_labels = dataset['test_labels']
	valid_dataset = dataset['validation_keys']
	valid_labels = dataset['validation_labels']
	
	del dataset
	train_dataset, train_labels = reformat(train_dataset, train_labels)
	test_dataset, test_labels = reformat(test_dataset, test_labels)
	valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)

	print('Training set   :', train_dataset.shape, train_labels.shape)
	print('Validation set : ', valid_dataset.shape, valid_labels.shape)
	print('Test set       : ', test_dataset.shape, test_labels.shape)
	time.sleep(5)

	createModel(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
	print "Done"
	return

def reformat(dataset, labels):	
	dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
	labels = (np.arange(num_classes) == labels[:, None]).astype(np.float32)
	return dataset, labels
	
def accuracy(output, target):
	true_positives = np.sum(np.argmax(output, 1) == np.argmax(target, 1))
	return (100.0 * true_positives / output.shape[0])
	
def computeModel( tf_sample_dataset ):		
	# Define convolution layer 1
	convolution = tf.nn.conv2d( tf_sample_dataset, weights_filter_1, [1, 1, 1, 1], padding='SAME' )
	pool_output = tf.nn.max_pool( tf.nn.relu(convolution + bias_filter_1), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
		
	# Define convolution layer 2
	convolution = tf.nn.conv2d( pool_output, weights_filter_2, [1, 1, 1, 1], padding='SAME' )
	pool_output = tf.nn.max_pool( tf.nn.relu(convolution + bias_filter_2), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

	# Flatten the output form convolution
	shape = pool_output.get_shape().as_list()
	flatten_activation = tf.reshape(pool_output, (shape[0], -1))
		
	# Define hidden layer 
	activation = tf.nn.relu(tf.matmul(flatten_activation, weights_hidden) + bias_hidden)
	# Dropout
	activation = tf.nn.dropout(activation, keep_prob)

	# Define output layer
	output = tf.matmul(activation, weights_output) + bias_output
	return output
	
def createModel(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
	print "Creating network graph"

	# Define input and target for the network
	tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size, image_size, num_channels))
	tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_classes))
	
	# Define validation and testing constants
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
		
	# Define convolution later 1 parameters
	global weights_filter_1
	global bias_filter_1
	weights_filter_1 = tf.Variable( tf.truncated_normal([patch_size, patch_size, num_channels, depth]) )
	bias_filter_1 = tf.Variable( tf.zeros([ depth ]) )
		
	# Define convolution later 2 parameters
	global weights_filter_2
	global bias_filter_2
	weights_filter_2 = tf.Variable( tf.truncated_normal([patch_size, patch_size, depth, depth]) )
	bias_filter_2 = tf.Variable( tf.zeros([ depth ]) )
		
	# Define hidden layer parameters
	global weights_hidden
	global bias_hidden
	weights_hidden = tf.Variable( tf.truncated_normal([ image_size/4 * image_size/4 * depth, hidden_size ]) )
	bias_hidden = tf.Variable( tf.zeros([ hidden_size ]) )
		
	# Define output layer parameters
	global weights_output
	global bias_output
	weights_output = tf.Variable( tf.truncated_normal([ hidden_size, num_classes ]) )
	bias_output = tf.Variable( tf.zeros([ num_classes ]) )
		
	output = computeModel( tf_train_dataset )	
		
	# Define loss and optimizer
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=output))
		
	# Regularization
	regularizer = tf.nn.l2_loss(weights_filter_1) + tf.nn.l2_loss(bias_filter_1) \
				+ tf.nn.l2_loss(weights_filter_2) + tf.nn.l2_loss(bias_filter_2) \
				+ tf.nn.l2_loss(weights_hidden) + tf.nn.l2_loss(bias_hidden) \
				+ tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(bias_output)

	loss = tf.reduce_mean(loss + beta * regularizer)
		
	# Define weight decay
	# global_step = tf.Variable(0)
	# dynamic_learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate)

	# Define optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
	# optimizer = tf.train.GradientDescentOptimizer(dynamic_learning_rate).minimize(loss, global_step=global_step)
		
	# Define predictions
	train_predictions = tf.nn.softmax(output)
		
	valid_predictions = tf.nn.softmax( computeModel(tf_valid_dataset) )
	test_predictions = tf.nn.softmax( computeModel(tf_test_dataset) )
	
	saver = tf.train.Saver()

	session = tf.Session()

	tf.global_variables_initializer().run(session=session)
	print "Graph Initialized"
	print ""
	time.sleep(5)
		
	for step in range(num_steps):
		offset = (step * batch_size) / (train_labels.shape[0] - batch_size)
		batch_dataset = train_dataset[offset : offset + batch_size, :]
		batch_labels = train_labels[offset : offset + batch_size, :]
			
		feed_dict = {tf_train_dataset :  batch_dataset, tf_train_labels : batch_labels}
		_, predictions = session.run([optimizer, train_predictions], feed_dict=feed_dict)
			
		if (step % 100) == 0:
			print "At step %04d, Batch accuracy = %.2f and Valid accuracy = %.2f" % \
				(step, accuracy(predictions, batch_labels), accuracy(valid_predictions.eval(session=session), valid_labels))
		
	print ""
	print "Test accuracy = %.2f" % (accuracy(test_predictions.eval(session=session), test_labels))
	print "Model training done"
	model = saver.save(session, model_path)
	session.close()
	print "Model saved at ", model
	return

if __name__ == "__main__":
	main()
	