#Author: Usama Munir Sheikh
#The following code implements a convolutional neural network classifier
	#for classifying images of objects 
	#in the CIFAR 10 dataset https://www.tensorflow.org/tutorials/deep_cnn#cifar-10_model
#It was written for my Intro to Deep Learning Course
	#taught by Professor Qiang Ji in Spring 2017

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import pickle
import cv2
#import cPickle
import random as rd
import time
from io import BytesIO
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML

image_number = 0

def showarray(a, fmt=str(image_number)+'.jpeg'):
	a = np.uint8(np.clip(a, 0, 1)*255)
	f = BytesIO()
	global image_number
	image_number = image_number + 1
	name = str(image_number)+'.jpeg'
	cv2.imwrite("weight_visualization/viz" + name, a)
    #PIL.Image.fromarray(a).save(f, fmt)
    #display(Image(data=f.getvalue()))
    
def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def render_naive(sess, t_obj, t_input, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj) 
    t_grad = tf.gradients(t_score, t_input)[0]
    
    img = np.random.uniform(size=(1, 32, 32, 3)) #+100
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input:img})
        # normalizing the gradient, so the same step size should work 
        g /= g.std()+1e-8         # for different layers and networks
        img += g*step
    clear_output()
    showarray(visstd(img[0,:,:,:]))

def main():
	time_initial = time.time()
	#Load Data
	with open('cifar_10_tf_train_test.pkl', 'rb') as f:
		u = pickle._Unpickler(f)
		u.encoding = 'latin1'
		train_x, train_y, test_x, test_y = u.load()
		f.close()

	#Convert to numpy array
	data_x = np.asarray(train_x)
	data_y = np.asarray(train_y)
	testdata_x = np.asarray(test_x)
	testdata_y = np.asarray(test_y)

	#Convert to Float
	data_x = data_x.astype(float)
	testdata_x = testdata_x.astype(float)

	#Divide by 255
	data_x = np.divide(data_x,255.0)
	testdata_x = np.divide(testdata_x,255.0)

	#Normalize (Subtract by Mean)
	mean_x = np.mean(data_x)
	testmean_x = np.mean(testdata_x)
	data_x = np.subtract(data_x, mean_x)
	testdata_x = np.subtract(testdata_x, testmean_x)

	#Variables for NN
	N = 50000 
	N_test = 5000;
	B = 128 
	K = 10 

	#Make Tensor Flow Variables and Placeholders
	X = tf.placeholder("float32",[None,32,32,3])
	Y = tf.placeholder("float32",[None])

	W1 = tf.get_variable("W1", shape=[5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
	b1 = tf.Variable(tf.zeros([32]))
	W2 = tf.get_variable("W2", shape=[5, 5, 32, 32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
	b2 = tf.Variable(tf.zeros([32]))
	W3 = tf.get_variable("W3", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
	b3 = tf.Variable(tf.zeros([64]))
	Wout = tf.get_variable("Wo", shape=[576, K], initializer=tf.contrib.layers.xavier_initializer_conv2d())
	bout = tf.Variable(tf.zeros([K]))

	eta = tf.Variable(0.0001)
	itrnum = tf.placeholder("int32")

	#Write Tensorflow equations and models
	#Forward Propagation
	conv1 = tf.nn.conv2d(X, W1, [1, 1, 1, 1], padding = 'VALID')
	C1 = tf.nn.bias_add(conv1, b1)
	A1 = tf.nn.relu(C1)
	pool1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

	conv2 = tf.nn.conv2d(pool1, W2, [1, 1, 1, 1], padding = 'VALID')
	C2 = tf.nn.bias_add(conv2, b2)
	A2 = tf.nn.relu(C2)
	pool2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

	conv3 = tf.nn.conv2d(pool2, W3, [1, 1, 1, 1], padding = 'VALID')
	C3 = tf.nn.bias_add(conv3, b3)
	A3 = tf.nn.relu(C3)

	batch_size = tf.shape(A3)[0]
	A3_flattened = tf.reshape(A3, [batch_size, -1])
	yout = tf.matmul(A3_flattened, Wout) + bout
	#print(yout.get_shape())

	#BackPropagation
	Y_int = tf.cast(Y, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_int, logits=yout)
	cost = tf.reduce_mean(cross_entropy)
	optimizer_step = tf.train.AdamOptimizer(learning_rate=eta).minimize(cost)
	
	#Adaptive Learning Rate
	'''
	eta_new = tf.train.exponential_decay(0.01,itrnum, 100,0.1,staircase=True)
	LRadapt = tf.assign(eta,eta_new)
	temp = eta
	'''

	#Accuracy Calculations
	yout_softmax = tf.nn.softmax(yout)
	predict_op = tf.argmax(yout_softmax,1)
	correct_prediction = tf.equal(predict_op, Y_int)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	# Create the collection.
	tf.get_collection("validation_nodes")
	#Add stuff to the collection.
	tf.add_to_collection("validation_nodes", X)
	tf.add_to_collection("validation_nodes", predict_op)
	#Save Model
	saver = tf.train.Saver()

	#Empty Matrices to save results
	loss_plot = []
	accuracy_train_plot = []
	accuracy_test_plot = []
	
	#Initialize model and Compute 
	num_itr = 100000
	model = tf.global_variables_initializer()	
	with tf.Session() as session:
		session.run(model)
		for i in range(num_itr):
			itr_number = i+1;
			
			#Pick Batch for Training
			indices_random = np.random.choice(N,B)
			data_X = data_x[indices_random]
			data_Y = data_y[indices_random]

			#Train
			loss_np = cost.eval(feed_dict={X: data_X, Y: data_Y})
			optimizer_step.run(feed_dict={X: data_X, Y: data_Y})

			#Adapt Learning Rate
			'''
			session.run(LRadapt,feed_dict={itrnum: itr_number})
			eta_np = temp.eval()
			'''

			if((itr_number % 200 == 0) or (itr_number == 1)):
				print('----------' + repr(i+1) + '----------')
				print(' ')
				#print('Learning Rate: ' + repr(eta_np))
				loss_plot.append(loss_np)
				print('Loss: ' + repr(loss_np))
				accuracy_train_np = accuracy.eval(feed_dict={X: data_X, Y: data_Y})
				accuracy_test_np = accuracy.eval(feed_dict={X:testdata_x,Y:testdata_y})
				accuracy_train_plot.append(accuracy_train_np*100)
				accuracy_test_plot.append(accuracy_test_np*100)
				print('Training Accuracy: ' + repr(accuracy_train_np*100))
				print('Test Accuracy: '+ repr(accuracy_test_np*100))
				print(' ')
				print('------------------------')

			if (accuracy_test_np*100) > 70.0:
				break

		#save session
		save_path = saver.save(session, "my_model")
		for n in range(32):
			render_naive(session, A1[:,:,:,n],X)

		#Get Predictions for Test Data
		testpredict_np = session.run(predict_op,feed_dict={X:testdata_x,Y:testdata_y})

	session.close()
	
	#Individual Accuracies
	testlabels = testdata_y.astype(int)
	totals = np.zeros([K,1])
	pred = np.zeros([K,1])
	ind_error = np.zeros([K,1])
	for i in range(N_test):
		y = testlabels[i]
		yo =  testpredict_np[i]
		totals[y,0] = totals[y,0] + 1.0
		if (y == yo):
			pred[y,0] = pred[y,0] + 1.0
	print(totals)
	print(pred)
	prederrors = np.subtract(totals,pred)
	ind_errors = np.divide(prederrors,totals)
	print('The Individual Errors are: ')
	print(ind_errors)

	#Print Elapsed Time
	elapsed = time.time() - time_initial
	print('Time Elapsed: ' + repr(elapsed))

	#print(np.shape(W1_np))
	itr_number = len(loss_plot)
	t = np.arange(itr_number)
	fig, ax1 = plt.subplots()
	ax1.plot(t,np.reshape(loss_plot,(itr_number,1)), 'b-')
	ax1.set_xlabel('Number of Iterations')
	ax1.set_ylabel('Loss', color='b')
	ax1.tick_params('y', colors='b')
	
	ax2 = ax1.twinx()
	ax2.plot(t,np.reshape(accuracy_train_plot,(itr_number,1)), 'r-')
	ax2.set_ylabel('Percent Accuracy', color='k')
	ax2.tick_params('y', colors='k')

	ax2.plot(t,np.reshape(accuracy_test_plot,(itr_number,1)), 'g-')
	
	fig.tight_layout()
	plt.show()
	
if __name__ == "__main__":
    main()