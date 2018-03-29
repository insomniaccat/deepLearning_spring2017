#Author: Usama Munir Sheikh
#The following code implements an LSTM recurrent neural network
	#for classifying tweets as positive or negative
	#in the sentiment140 dataset http://help.sentiment140.com/for-students/
#It was written for my Intro to Deep Learning Course
	#taught by Professor Qiang Ji in Spring 2017

import tensorflow as tf
import numpy as np
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time

def main():
	time_initial = time.time() #To see how much time required for the entire code to run

	#Load Training and Validation Data
	npzfile = np.load("train_and_val.npz")
	train_x = npzfile["train_x"]
	train_y = npzfile["train_y"]
	train_mask = npzfile["train_mask"]
	val_x = npzfile["val_x"]
	val_y = npzfile["val_y"]
	val_mask = npzfile["val_mask"]

	#Parameters
	N = 400000
	N_val = 50000;
	B = 1000 #batch_size

	#Network Parameters
	max_sequence_length = 25
	vocab_size = 8745
	word_embedding_size = 300
	cell_size = 128 #rnn cell size
	eta = 0.001 #learning rate

	#Make Tensor Flow Variables and Placeholders
	X = tf.placeholder(tf.int32,[None,max_sequence_length]) 
	Y = tf.placeholder(tf.float32,[None])
	Mask = tf.placeholder(tf.int32,[None,max_sequence_length]) 
	w_embed = tf.Variable(tf.random_uniform([vocab_size, word_embedding_size], minval=-0.1, maxval=0.1, seed = 1230))
	W = tf.get_variable("W", shape=[cell_size, 1], initializer=tf.contrib.layers.xavier_initializer())
	b = tf.Variable(tf.zeros([1]))

	#Write Tensorflow equations and models
	rnn_input = tf.nn.embedding_lookup(w_embed, X) #Word Embedding
	
	cell = tf.nn.rnn_cell.LSTMCell(cell_size) #create LSTM rnn cell
	output, state = tf.nn.dynamic_rnn(cell, rnn_input, dtype=tf.float32, time_major=False) #Propagate through rnn cell

	#Masking
	length = tf.cast(tf.reduce_sum(Mask,reduction_indices=1), tf.int32)
	batch_size = tf.shape(X)[0]
	max_length = tf.shape(output)[1]
	out_size = int(output.get_shape()[2])
	flat = tf.reshape(output, [-1, out_size])
	index = tf.range(0, batch_size)*max_length + (length - 1)
	relevant = tf.gather(flat, index)

	yout = tf.matmul(relevant, W) + b #estimated output
	Y_reshaped = tf.reshape(Y, [batch_size,1])
	cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=yout, labels=Y_reshaped)
	cost = tf.reduce_mean(cross_entropy) #cross entropy cost
	optimizer_step = tf.train.AdamOptimizer(learning_rate=eta).minimize(cost) #run optimizer
	
	#Accuracy Calculations
	Y_int = tf.cast(Y_reshaped, tf.int64)
	yout_sigmoid = tf.nn.sigmoid(yout)
	predict_op = tf.cast(tf.round(yout_sigmoid), tf.int64)
	correct_prediction = tf.equal(predict_op, Y_int)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	
	# Create the collection.
	tf.get_collection("validation_nodes")
	#Add stuff to the collection.
	tf.add_to_collection("validation_nodes", X)
	tf.add_to_collection("validation_nodes", Mask)
	tf.add_to_collection("validation_nodes", predict_op)
	#Save Model
	saver = tf.train.Saver()

	#Create Empty Matrices to Save results
	loss_plot = []
	accuracy_train_plot = []
	accuracy_test_plot = []
	n = 50
	num_epochs = 10 #number of epochs
	num_itr = int(np.divide(N,B)) #number of iterations per epoch
	model = tf.global_variables_initializer()	
	with tf.Session() as session:
		session.run(model)
		for j in range(num_epochs):
			epoch_number = j+1
			for i in range(num_itr):
				itr_number = i+1
				#Pick Batch for Training
				indices = np.arange(B) + (i*B)
				data_X = train_x[indices]
				data_Y = train_y[indices]
				mask = train_mask[indices]

				#Train
				loss_np = cost.eval(feed_dict={X: data_X, Y: data_Y, Mask: mask})
				optimizer_step.run(feed_dict={X: data_X, Y: data_Y, Mask: mask})

				#Accuracy #PrintValues # SaveResults #EveryFiftyIterations

				if((itr_number % n == 0) or (itr_number == 1)):
					print('----------' + repr(i+1) + '----------')
					print(' ')

					#print('Learning Rate: ' + repr(eta_np))

					loss_plot.append(loss_np)
					print('Loss: ' + repr(loss_np))

					accuracy_train_np = accuracy.eval(feed_dict={X: data_X, Y: data_Y, Mask: mask}) #training accuracy
					accuracy_test_np = accuracy.eval(feed_dict={X:val_x,Y:val_y, Mask: val_mask}) #validation accuracy

					accuracy_train_plot.append(accuracy_train_np*100)
					accuracy_test_plot.append(accuracy_test_np*100)

					print('Training Accuracy: ' + repr(accuracy_train_np*100))
					print('Test Accuracy: '+ repr(accuracy_test_np*100))

					print(' ')
					print('------------------------')
					if (accuracy_test_np*100) > 84.1:
						break
			if (accuracy_test_np*100) > 84.1:
				break

		word_embedding_matrix = w_embed.eval() #save word embedding matrix for visualization
		#save session
		save_path = saver.save(session, "my_model")
	session.close()

	#Print Elapsed Time
	print('------------------------')
	print('Optimization Finished')
	elapsed = time.time() - time_initial
	print('Time Elapsed: ' + repr(elapsed))

	#Visualization
	with open("vocab.json", "r") as f:
		vocab = json.load(f)
	s = ["monday", "tuesday", "wednesday", "thursday", "friday",
		"saturday", "sunday", "orange", "apple", "banana", "mango",
		"pineapple", "cherry", "fruit"]
	words = [(i, vocab[i]) for i in s]

	model = TSNE(n_components=2, random_state=0)
	#Note that the following line might use a good chunk of RAM
	tsne_embedding = model.fit_transform(word_embedding_matrix)
	words_vectors = tsne_embedding[np.array([item[1][0] for item in
		words])]

	z = words_vectors[:,0] #x-axis
	y = words_vectors[:,1] #y-axis
	fig, ax = plt.subplots()
	ax.scatter(z, y)

	for i, txt in enumerate(s):
		ax.annotate(txt, (z[i],y[i]))

	plt.show()

	#Plots
	itr_number = len(loss_plot)
	t = np.arange(itr_number)
	fig, ax1 = plt.subplots()
	ax1.plot(t,np.reshape(loss_plot,(itr_number,1)), 'b-')
	ax1.set_xlabel('Number of Iterations (pghw5)')
	ax1.set_ylabel('Loss', color='b')
	ax1.tick_params('y', colors='b')
	
	ax2 = ax1.twinx()
	ax2.plot(t,np.reshape(accuracy_train_plot,(itr_number,1)), 'r-')
	ax2.set_ylabel('Percent Accuracy (pghw5)', color='k')
	ax2.tick_params('y', colors='k')

	ax2.plot(t,np.reshape(accuracy_test_plot,(itr_number,1)), 'g-')
	
	fig.tight_layout()
	plt.show()

if __name__ == "__main__":
    main()