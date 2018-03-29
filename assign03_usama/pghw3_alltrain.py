#Author: Usama Munir Sheikh
#The following code implements back propagation in a neural network
	#of 2 hidden layers of 100 nodes each
	#for classifying digits 1 through 10
	#in the MNIST dataset https://en.wikipedia.org/wiki/MNIST_database
#It was written for my Intro to Deep Learning Course
	#taught by Professor Qiang Ji in Spring 2017

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pickle
import random as rd

def one_of_K_coding(array, num_classes):
	array_size = len(array)
	array = np.transpose(array.astype(int))
	array_one_of_K = np.zeros((array_size,num_classes))
	array_one_of_K[np.arange(array_size), array] =1

	return array_one_of_K

def main():

	#Load Training Data Labels
	filename = "labels/train_label.txt"
	data_y = np.loadtxt(filename,delimiter=' ')
	N = len(data_y) #Number of images
	data_y = np.reshape(data_y,(N,1))
	#print(data_y.shape)
	#print(data_y)

	#Load Training Data Images
	path = "train_data/"
	imlist = os.listdir(path)
	imlist.sort()
	filenames = [];
	for file in imlist: #loop for all files in folder
		filenames.append(path+file)
	image = mpimg.imread(filenames[0])
	vector_image = image.reshape(-1)
	M = len(vector_image) #dimension of image (784)
	#print(M)
	data_x = np.zeros((N,M))
	for i in range(0,N): #for 0 to M-1
		image = mpimg.imread(filenames[i])
		vector_image = image.reshape(-1)
		vector_image_normalized = np.divide(vector_image,255.0)
		vector_image_normalized = np.reshape(vector_image_normalized,(1,M))
		data_x[i,:] = vector_image_normalized
	#one_vector = np.ones((N,1))
	#data_x = np.append(data_x,one_vector,axis=1)

	#load test data
	filename_test = "labels/test_label.txt"
	data_y_test = np.loadtxt(filename_test,delimiter=' ')
	N_test = len(data_y_test) #Number of images
	data_y_test = np.reshape(data_y_test,(N_test,1))

	#Load Test Data Images
	path_test = "test_data/"
	imlist_test = os.listdir(path)
	imlist_test.sort()
	filenames_test = [];
	for file_test in imlist_test: #loop for all files in folder
		filenames_test.append(path_test+file_test)
	image_test = mpimg.imread(filenames_test[0])
	vector_image_test = image_test.reshape(-1)
	M_test = len(vector_image_test) #dimension of image (784)
	#print(M)
	data_x_test = np.zeros((N_test,M_test))
	for i in range(0,N_test): #for 0 to M-1
		image_test = mpimg.imread(filenames_test[i])
		vector_image_test = image_test.reshape(-1)
		vector_image_normalized_test = np.divide(vector_image_test,255.0)
		vector_image_normalized_test = np.reshape(vector_image_normalized_test,(1,M_test))
		data_x_test[i,:] = vector_image_normalized_test

	#Variables for NN
	M1 = M #Nodes in first layer 
	M2 = 100#Number of nodes in second layer
	M3 = 100 #Number of nodes in third layer 
	M4 = 10 #Nodes in last layer
	B = 50 #batch size
	K = M4 #Number of classes
	eta = 0.1
	data_y_coded = one_of_K_coding(data_y, K)
	data_y_coded = np.transpose(data_y_coded)
	data_x = np.transpose(data_x)

	data_y_coded_test = one_of_K_coding(data_y_test, K)
	data_y_coded_test = np.transpose(data_y_coded_test)
	data_x_test = np.transpose(data_x_test)

	#Make Tensor Flow Variables and Placeholders
	X = tf.placeholder("float32",[M,B])
	Y = tf.placeholder("float32",[K,B])
	X_test = tf.placeholder("float32",[M,N_test])
	Y_test = tf.placeholder("float32",[K,N_test])
	X_train = tf.placeholder("float32",[M,N])
	Y_train = tf.placeholder("float32",[K,N])
	W1 = tf.Variable(tf.random_uniform([M1,M2], minval=-0.1, maxval=0.1, seed = 1230))
	W10 = tf.Variable(tf.zeros([M2,1]))
	W2 = tf.Variable(tf.random_uniform([M2,M3], minval=-0.1, maxval=0.1, seed = 2052))
	W20 = tf.Variable(tf.zeros([M3,1]))
	W3 = tf.Variable(tf.random_uniform([M3,M4], minval=-0.1, maxval=0.1, seed = 134))
	W30 = tf.Variable(tf.zeros([M4,1]))
	#eta = tf.Variable(tf.constant(0.5))

	#Write Tensorflow equations and models
	#Forward Propagation
	W10_expand = tf.tile(W10, [1,B])
	H1_z = tf.add(tf.matmul(tf.transpose(W1),X),W10_expand)
	H1 = tf.nn.relu(tf.transpose(tf.expand_dims(H1_z,0)))

	W20_expand = tf.tile(W20, [1,B])
	H2_z = tf.add(tf.matmul(tf.transpose(W2),H1_z),W20_expand)
	H2 = tf.nn.relu(tf.transpose(tf.expand_dims(H2_z,0)))
	
	W30_expand = tf.tile(W30, [1,B])
	yout_s = tf.add(tf.matmul(tf.transpose(W3),H2_z),W30_expand)
	exp_yout = tf.exp(yout_s)
	exp_yout_sum = tf.reshape(tf.reduce_sum(exp_yout, 0),[1,B])
	temporary = tf.tile(exp_yout_sum,[10,1])
	yout = tf.div(exp_yout,temporary)
	del_yout = tf.multiply(-1.0,tf.subtract(Y,yout))
	
	sq_sub = tf.subtract(Y,yout)
	sq_sub_expand = tf.expand_dims(tf.transpose(sq_sub),1)
	sq_sub_expand_2 = tf.transpose(tf.expand_dims(sq_sub,0))
	sq = tf.reduce_sum(tf.matmul(sq_sub_expand,sq_sub_expand_2),0)
	squared_loss = tf.squeeze(tf.multiply(0.02, tf.multiply(0.5,sq)))

	#Back Propagation
	#Output Layer
	yout_expand =  tf.transpose(tf.expand_dims(yout,0))
	del_yout_expand = tf.transpose(tf.expand_dims(del_yout,0))
	term1 = tf.reduce_sum(tf.multiply(yout,del_yout),0)
	term1_1 = tf.transpose(tf.expand_dims(tf.expand_dims(term1,0),0))
	term1_2 = tf.tile(term1_1,[1,K,1])
	term2 = tf.multiply(yout_expand,tf.subtract(del_yout_expand,term1_2))
	term3 = tf.reshape(term2,[B,1,K])
	term4 = tf.tile(H2,[1,1,K])
	term5 = tf.multiply(term4,term3)
	delW3 = tf.multiply(0.02,tf.reduce_sum(term5,0))
	
	delW30 = tf.multiply(0.02,tf.reduce_sum(term2,0))
	
	temp0 = tf.reshape(yout_expand,[B,1,K]) #y expanded
	temp1 = tf.multiply(temp0,W3)
	temp2 = tf.reduce_sum(temp1,2)
	temp2_1 = tf.reshape(tf.expand_dims(temp2,1),[B,M3,1])
	temp3 = tf.tile(temp2_1,[1,1,K])
	temp4 = tf.subtract(W3,temp3) #inside term(W-sum(yW))
	temp5 = tf.reshape(del_yout_expand,[B,1,K])#dely expanded
	temp6 = tf.multiply(temp0,temp4)
	temp7 = tf.multiply(temp5,temp6)
	temp8 = tf.reduce_sum(temp7,2)
	delH2 = tf.reshape(tf.expand_dims(temp8,1),[B,M3,1])

	#Second Hidden Layer
	tor0 = tf.constant(0, dtype=tf.float32)
	tor1 = tf.not_equal(H2, tor0)
	tor2 = tf.to_float(tor1)
	tor3 = tf.reshape(tor2,[B,1,M3])
	tor4 = tf.reshape(delH2,[B,1,M3])
	tor5 = tf.multiply(tor3,tor4)
	deriv = tf.tile(H1,[1,1,M3])
	loco = tf.multiply(deriv,tor5)
	delW2 = tf.multiply(0.02,tf.reduce_sum(loco,0))

	delW20 = tf.multiply(0.02,tf.transpose(tf.reduce_sum(tor5,0)))
	
	tor6 = tf.tile(tf.expand_dims(W2,0),[B,1,1])
	tor7 = tf.multiply(tor3,tor4)
	tor8 = tf.multiply(tor6,tor7)
	tor9 = tf.reduce_sum(tor8,2)
	delH1 = tf.reshape(tf.expand_dims(tor9,1),[B,M2,1])

	#First Hidden Layer
	toro0 = tf.constant(0, dtype=tf.float32)
	toro1 = tf.not_equal(H1, toro0)
	toro2 = tf.to_float(toro1)
	toro3 = tf.reshape(toro2,[B,1,M2])
	toro4 = tf.reshape(delH1,[B,1,M2])
	toro5 = tf.multiply(toro3,toro4)
	X_expand = tf.transpose(tf.expand_dims(X,0))
	deriva = tf.tile(X_expand,[1,1,M2])
	locoo = tf.multiply(deriva,toro5)
	delW1 = tf.multiply(0.02,tf.reduce_sum(locoo,0))

	delW10 = tf.multiply(0.02,tf.transpose(tf.reduce_sum(toro5,0)))
	
	#Update Weights
	new_W3 = tf.subtract(W3,tf.multiply(eta,delW3))
	new_W30 = tf.subtract(W30,tf.multiply(eta,delW30))
	Update_W3 = tf.assign(W3,new_W3)
	Update_W30 = tf.assign(W30,new_W30)

	new_W2 = tf.subtract(W2,tf.multiply(eta,delW2))
	new_W20 = tf.subtract(W20,tf.multiply(eta,delW20))
	Update_W2 = tf.assign(W2,new_W2)
	Update_W20 = tf.assign(W20,new_W20)

	new_W1 = tf.subtract(W1,tf.multiply(eta,delW1))
	new_W10 = tf.subtract(W10,tf.multiply(eta,delW10))
	Update_W1 = tf.assign(W1,new_W1)
	Update_W10 = tf.assign(W10,new_W10)

	#new_eta = tf.minimum(0.4,squared_loss)
	#new_eta = tf.minimum(0.4,0.01)
	#Update_eta = tf.assign(eta,new_eta)
	
	#Forward Prop for Test Data
	W10_expand_train = tf.tile(W10, [1,N])
	H1_z_train = tf.add(tf.matmul(tf.transpose(W1),X_train),W10_expand_train)
	H1_train = tf.nn.relu(tf.transpose(tf.expand_dims(H1_z_train,0)))

	W20_expand_train = tf.tile(W20, [1,N])
	H2_z_train = tf.add(tf.matmul(tf.transpose(W2),H1_z_train),W20_expand_train)
	H2_train = tf.nn.relu(tf.transpose(tf.expand_dims(H2_z_train,0)))
	
	W30_expand_train = tf.tile(W30, [1,N])
	yout_s_train = tf.add(tf.matmul(tf.transpose(W3),H2_z_train),W30_expand_train)
	exp_yout_train = tf.exp(yout_s_train)
	exp_yout_sum_train = tf.reshape(tf.reduce_sum(exp_yout_train, 0),[1,N])
	temporary_train = tf.tile(exp_yout_sum_train,[10,1])
	yout_train = tf.div(exp_yout_train,temporary_train)

	#Forward Prop for Test Data
	W10_expand_test = tf.tile(W10, [1,N_test])
	H1_z_test = tf.add(tf.matmul(tf.transpose(W1),X_test),W10_expand_test)
	H1_test = tf.nn.relu(tf.transpose(tf.expand_dims(H1_z_test,0)))

	W20_expand_test = tf.tile(W20, [1,N_test])
	H2_z_test = tf.add(tf.matmul(tf.transpose(W2),H1_z_test),W20_expand_test)
	H2_test = tf.nn.relu(tf.transpose(tf.expand_dims(H2_z_test,0)))
	
	W30_expand_test = tf.tile(W30, [1,N_test])
	yout_s_test = tf.add(tf.matmul(tf.transpose(W3),H2_z_test),W30_expand_test)
	exp_yout_test = tf.exp(yout_s_test)
	exp_yout_sum_test = tf.reshape(tf.reduce_sum(exp_yout_test, 0),[1,N_test])
	temporary_test = tf.tile(exp_yout_sum_test,[10,1])
	yout_test = tf.div(exp_yout_test,temporary_test)


	num_itr = 7400
	y_int = data_y_test.astype(int)
	y_int_train = data_y.astype(int)
	loss_plot = []
	error_train_plot = []
	error_test_plot = []
	flag = 0
	#Initialize model and Compute 
	model = tf.global_variables_initializer()	
	with tf.Session() as session:
		session.run(model)
		for i in range(num_itr):
			print(i)
			indices_random = np.random.choice(N,B)
			data_X = data_x[:,indices_random]
			data_Y = data_y_coded[:,indices_random]

			squared_loss_np,W1_np,W10_np,W2_np,W20_np,W3_np,W30_np,yout_train_np,yout_test_np = session.run([squared_loss,Update_W1,Update_W10,Update_W2,Update_W20,Update_W3,Update_W30,yout_train,yout_test],feed_dict={X: data_X, Y: data_Y, X_test:data_x_test,Y_test:data_y_coded_test, X_train:data_x,Y_train:data_y_coded})
			
			value_train = np.argmax(np.transpose(yout_train_np), axis = 1)
			value_train = np.reshape(value_train, (N, 1))
			
			value_test = np.argmax(np.transpose(yout_test_np), axis = 1)
			value_test = np.reshape(value_test, (N_test, 1))
			
			accuracySum_train = np.sum(value_train == y_int_train)
			accuracySum_test = np.sum(value_test == y_int)

			#print(accuracySum_test)
			Accuracy_train = (np.divide(np.float32(accuracySum_train),np.float32(N)))*100.0
			Accuracy_test = (np.divide(np.float32(accuracySum_test),np.float32(N_test)))*100.0

			error_train = 1.0 - (np.divide(np.float32(accuracySum_train),np.float32(N)))
			error_test = 1.0 - (np.divide(np.float32(accuracySum_train),np.float32(N)))

			loss_plot.append(squared_loss_np)
			error_train_plot.append(error_train)
			error_test_plot.append(error_test)
			print(squared_loss_np)
			print(Accuracy_train)
			print(Accuracy_test)
			itr = i
			
			if (Accuracy_train > 88.0)  and (flag == 0):
				flag = 1
			'''
			elif (Accuracy_train > 70.0)  and (flag == 1):
				flag = 2
			elif (Accuracy_train > 60.0) and (flag == 0):
				flag = 1
			'''
			if flag == 1:
				eta = 0.001
			'''
			elif flag == 2:
				eta = 0.01
			elif flag == 1:
				eta = 0.1

			if Accuracy_train > 95.0:
				break
			print(eta)
			'''
			#print(squared_loss_np)
			#print(result1)
			#print("apple")
			#print(result2)
	session.close()
	
	Theta = [W1_np, W10_np, W2_np, W20_np, W3_np, W30_np]
	filehandler = open("nn_parameters.txt","wb")
	pickle.dump(Theta, filehandler, protocol=2)
	filehandler.close()

	itr = itr+1
	t = np.arange(itr)

	plt.figure()
	plt.plot(t,np.reshape(loss_plot,(itr,1)))
	plt.title(r'Squared Loss')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Value of Loss Function')
	#plt.show()

	plt.figure()
	plt.plot(t,np.reshape(error_train_plot,(itr,1)))
	plt.title(r'Average Training Classification Error')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Training Error')
	#plt.show()

	plt.figure()
	plt.plot(t,np.reshape(error_test_plot,(itr,1)))
	plt.title(r'Average Test Classification Error')
	plt.xlabel('Number of Iterations')
	plt.ylabel('Test Error')
	plt.show()


if __name__ == "__main__":
    main()