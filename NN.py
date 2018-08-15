import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import tensorflow as tf
import cPickle as pickle
import time

dataset_file = '/home/padmanabh/Work/notMNIST.pickle'
model_path = '/home/padmanabh/Work/model.ckpt'

image_size = 28
num_classes = 10

# learning parameters
batch_size = 1024
hidden_size = 1024
num_steps = 3001
learning_rate = 0.05

# parameters for weight decay
decay_steps = 100
decay_rate = 0.9

# parameters for regularization
beta = 0.05

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
	print('Training set   :', train_dataset.shape, train_labels.shape)
	print('Validation set : ', valid_dataset.shape, valid_labels.shape)
	print('Test set       : ', test_dataset.shape, test_labels.shape)
	
	train_dataset, train_labels = reformat(train_dataset, train_labels)
	test_dataset, test_labels = reformat(test_dataset, test_labels)
	valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)

	createModel(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
	print "Done"
	return

def reformat(dataset, labels):	
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	labels = (np.arange(num_classes) == labels[:, None]).astype(np.float32)
	return dataset, labels
	
def accuracy(output, target):
	true_positives = np.sum(np.argmax(output, 1) == np.argmax(target, 1))
	return (100.0 * true_positives / output.shape[0])
	
def createModel(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
	graph = tf.Graph()
	print "Creating network graph"
	with graph.as_default():
		# Define input and target for the network
		tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size * image_size))
		tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_classes))
		
		# Define validation and testing constants
		tf_valid_dataset = tf.constant(valid_dataset)
		tf_test_dataset = tf.constant(test_dataset)
		
		# Define hidden layer parameters
		weights_hidden = tf.Variable( tf.truncated_normal([ image_size * image_size, hidden_size ]) )
		bias_hidden = tf.Variable( tf.zeros([ hidden_size ]) )
		
		# Define output layer parameters
		weights_output = tf.Variable( tf.truncated_normal([ hidden_size, num_classes ]) )
		bias_output = tf.Variable( tf.zeros([ num_classes ]) )

		# Define hidden layer 
		activation = tf.nn.relu(tf.matmul(tf_train_dataset, weights_hidden) + bias_hidden)
		# Dropout
		activation = tf.nn.dropout(activation, keep_prob)
		
		# Define output layer
		output = tf.matmul(activation, weights_output) + bias_output
		
		# Define loss and optimizer
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=output))
		
		# Regularization
		regularizer = tf.nn.l2_loss(weights_hidden) + tf.nn.l2_loss(weights_output)
		loss = tf.reduce_mean(loss + beta * regularizer)
		
		# Define weight decay
		global_step = tf.Variable(0)
		dynamic_learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate)

		# Define optimizer
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
		optimizer = tf.train.GradientDescentOptimizer(dynamic_learning_rate).minimize(loss, global_step=global_step)
		
		# Define predictions
		train_predictions = tf.nn.softmax(output)
		
		valid_activation = tf.nn.relu(tf.matmul(valid_dataset, weights_hidden) + bias_hidden)
		valid_predictions = tf.nn.softmax(tf.matmul(valid_activation, weights_output) + bias_output)
		
		test_activation = tf.nn.relu(tf.matmul(test_dataset, weights_hidden) + bias_hidden)
		test_predictions = tf.nn.softmax(tf.matmul(test_activation, weights_output) + bias_output)
		
		saver = tf.train.Saver()

	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run()
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
					(step, accuracy(predictions, batch_labels), accuracy(valid_predictions.eval(), valid_labels))
		
		print ""
		print "Test accuracy = %.2f" % (accuracy(test_predictions.eval(), test_labels))
		print "Model training done"
		model = saver.save(session, model_path)
		print "Model saved at ", model
	return

if __name__ == "__main__":
	main()
	