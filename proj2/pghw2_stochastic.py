#Author: Usama Munir Sheikh
#The following code implements a multi-class logistic regressor
	#for classifying digits 1 through 5 
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

def main():
	
	#Load Training Data Labels
	filename = "labels/train_label.txt"
	data_y = np.loadtxt(filename,delimiter=' ')
	N = len(data_y)
	data_y = np.reshape(data_y,(N,1))

	#Load Training Data Images
	path = "train_data/"
	imlist = os.listdir(path)
	imlist.sort()
	filenames = [];
	for file in imlist: #loop for all files in folder
		filenames.append(path+file)
	image = mpimg.imread(filenames[0])
	vector_image = image.reshape(-1)
	pixel_num = len(vector_image)
	data_x = np.zeros((N,pixel_num))
	for i in range(0,N): #for 0 to M-1
		image = mpimg.imread(filenames[i])
		vector_image = image.reshape(-1)
		vector_image_normalized = np.divide(vector_image,255.0)
		vector_image_normalized = np.reshape(vector_image_normalized,(1,pixel_num))
		data_x[i,:] = vector_image_normalized
	one_vector = np.ones((N,1))
	data_x = np.append(data_x,one_vector,axis=1)
	imsize = pixel_num +1

	M = 100 #batchsize
	sizo = np.int(N/M) #number of batches
	print(sizo)

	#Make Tensor Flow Variables and Placeholders
	K = tf.constant(5)
	lamb = tf.constant(0.02)
	X = tf.placeholder("float32",[M,imsize])
	Y = tf.placeholder("float32",[M,1])
	W = tf.Variable(0.1*tf.ones([K,imsize]))
	eta = tf.constant(0.0055) #learningrate
	
	#Write Tensorflow equations and models
	num = tf.exp(tf.matmul(X, W,transpose_a=False, transpose_b=True)) #numerator
	den = tf.reshape(tf.reduce_sum(num,1),[M,1])
	den_expand = tf.tile(den, [1,K])
	temp0 = tf.div(num,den_expand)
	sigma = tf.log(temp0)
	row_indices = tf.reshape(tf.range(M),[1,M])
	col_indices = tf.cast(tf.subtract(tf.reshape(Y,[1,M]),tf.ones([1,M])),tf.int32)
	coords = tf.squeeze(tf.transpose(tf.stack([row_indices, col_indices])))
	temp1 = tf.zeros((M,K))
	Ymk = tf.sparse_to_dense(coords, temp1.get_shape(), 1.0)
	temp2 =tf.multiply(Ymk,sigma)
	Loss = tf.multiply(-1.0,tf.reduce_sum(temp2))

	temp3 = tf.subtract(Ymk,temp0)
	temp4 = tf.unstack(temp3, num=5, axis=1)
	Ysigma1 = tf.reshape(temp4[0],[M,1])
	Ysigma2 = tf.reshape(temp4[1],[M,1])
	Ysigma3 = tf.reshape(temp4[2],[M,1])
	Ysigma4 = tf.reshape(temp4[3],[M,1])
	Ysigma5 = tf.reshape(temp4[4],[M,1])
	Ysigma1_expand = tf.tile(Ysigma1, [1,imsize])
	Ysigma2_expand = tf.tile(Ysigma2, [1,imsize])
	Ysigma3_expand = tf.tile(Ysigma3, [1,imsize])
	Ysigma4_expand = tf.tile(Ysigma4, [1,imsize])
	Ysigma5_expand = tf.tile(Ysigma5, [1,imsize])
	theta1 = tf.multiply(X,Ysigma1_expand)
	theta2 = tf.multiply(X,Ysigma2_expand)
	theta3 = tf.multiply(X,Ysigma3_expand)
	theta4 = tf.multiply(X,Ysigma4_expand)
	theta5 = tf.multiply(X,Ysigma5_expand)
	grad1 = tf.reduce_sum(theta1,0)
	grad2 = tf.reduce_sum(theta2,0)
	grad3 = tf.reduce_sum(theta3,0)
	grad4 = tf.reduce_sum(theta4,0)
	grad5 = tf.reduce_sum(theta5,0)
	temporary = tf.multiply(0.5,W)

	reg = tf.multiply(lamb,temporary)

	temp5 = tf.stack([grad1, grad2, grad3, grad4, grad5])
	temp6 = tf.add(temp5,reg)
	Gradient = tf.multiply(-1.0,temp6)
	New_W = tf.subtract(W,tf.multiply(eta,Gradient))
	Update_W = tf.assign(W,New_W)

	##Initialize model and Compute 
	itr = 1000
	model = tf.global_variables_initializer()	
	with tf.Session() as session:
		session.run(model)
		for i in range(itr):
			batchnum = rd.randint(0,sizo-1)
			flag = True
			while (flag == True):
				if (batchnum*M >= 0 and (batchnum*M)+M < N):
					flag = False
				else:
					batchnum = rd.randint(0,sizo-1)
			#print(batchnum)
			#batchx = data_x[(batchnum*M):(batchnum*M)+M]
			#batchy = data_y[(batchnum*M):(batchnum*M)+M]
			data_X = data_x[(batchnum*M):(batchnum*M)+M]
			data_Y = data_y[(batchnum*M):(batchnum*M)+M]
			loss = session.run(Loss,feed_dict={X: data_X, Y: data_Y})
			Wmatrix = session.run(Update_W,feed_dict={X: data_X, Y: data_Y})
			#print(i)
			#print(Wmatrix)
			#print(loss)
			#result = session.run(update_W,feed_dict={W_temp: temp_W})
		print("Final Loss Value: ")
		print(loss)
		print("Wmatrix: ")
		print(np.transpose(Wmatrix))
		#print(Wmatrix.shape)
	session.close()	

	#Output Data in File
	#filehandler = open("multiclass_parameters.txt","wb")
	#pickle.dump(np.transpose(Wmatrix), filehandler)
	#filehandler.close()
	filehandler = "multiclass_parameters.txt"
	np.savetxt(filehandler, np.transpose(Wmatrix),delimiter=' ')

	#*************TEST DATA SET***********#
	#Load Test Data Labels
	filename = "labels/test_label.txt"
	data_y = np.loadtxt(filename,delimiter=' ')
	M = len(data_y)
	data_y = np.reshape(data_y,(M,1))
	#print(data_y.shape)
	#print(data_y)

	#Load Training Data Images
	path = "test_data/"
	imlist = os.listdir(path)
	imlist.sort()
	filenames = [];
	for file in imlist: #loop for all files in folder
		filenames.append(path+file)
	image = mpimg.imread(filenames[0])
	vector_image = image.reshape(-1)
	pixel_num = len(vector_image)
	data_x = np.zeros((M,pixel_num))
	for i in range(0,M): #for 0 to M-1
		image = mpimg.imread(filenames[i])
		vector_image = image.reshape(-1)
		vector_image_normalized = np.divide(vector_image,255.0)
		vector_image_normalized = np.reshape(vector_image_normalized,(1,pixel_num))
		data_x[i,:] = vector_image_normalized
	one_vector = np.ones((M,1))
	data_x = np.append(data_x,one_vector,axis=1)
	imsize = pixel_num +1
	accuracySum = 0
	for j in range(M):
		sigma = [0,0,0,0,0]
		for k in range(5):
			numer = np.exp(np.dot(Wmatrix[k,:],data_x[j,:]))
			denom = np.sum(np.exp(np.matmul(data_x[j,:],np.transpose(Wmatrix))))
			sigma[k] = numer/denom
		value = np.argmax(sigma) + 1
		#print("value: ")
		#print(value)
		#print("actual")
		#print(np.int(data_y[j]))
		if (np.int(data_y[j]) == value):
			accuracySum = accuracySum + 1

	Accuracy = (np.divide(np.float32(accuracySum),np.float32(M)))*100.0
	print("Accuracy Percentage on Test Data Set: ")
	print(Accuracy)
	
	#Plots
	fig = plt.figure()
	plt.subplot(161)
	W_k = Wmatrix[0,0:imsize-1]
	plt.imshow(W_k.reshape(28,28))
	plt.subplot(162)
	W_k = Wmatrix[1,0:imsize-1]
	plt.imshow(W_k.reshape(28,28))
	plt.subplot(163)
	W_k = Wmatrix[2,0:imsize-1]
	plt.imshow(W_k.reshape(28,28))
	plt.subplot(164)
	W_k = Wmatrix[3,0:imsize-1]
	plt.imshow(W_k.reshape(28,28))
	plt.subplot(165)
	W_k = Wmatrix[4,0:imsize-1]
	plt.imshow(W_k.reshape(28,28))

	#plt.colorbar()
	plt.show()


if __name__ == "__main__":
    main()
