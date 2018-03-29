#Author: Usama Munir Sheikh
#The following code implements the analytical solution
	#for linear regression using least squares
	#finds weights
#It was written for my Intro to Deep Learning Course
	#taught by Professor Qiang Ji in Spring 2017

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def main():
	#Load Data
	filename = "Prog1_data.txt"
	data = np.loadtxt(filename,delimiter=' ')
	[N,M] = data.shape
	data_x = data[0:N,0:M-1]
	one_vector = np.ones((N,1)) #Create a column vector of ones
	data_x = np.append(data_x,one_vector,axis=1) #Create matrix X
	data_y = data[0:N,-1:]; #Y

	#Make Tensor Flow Variables and Placeholders
	X = tf.placeholder("float32",[N,M])
	Y = tf.placeholder("float32",[N,1])
	W = tf.Variable(tf.zeros([M,1]))

	#Write Tensorflow equations and models
	psuedo = tf.matmul(tf.matrix_inverse(tf.matmul(X,X, transpose_a=True)),X, transpose_a=False, transpose_b=True)
	theta = tf.matmul(psuedo,Y)

	##Initialize model and Compute 
	model = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(model)
		result = session.run(theta, feed_dict={X: data_x, Y: data_y})
		print("Theta:\n", result)
		Xtimestheta = np.matmul(data_x,result)
		Loss = (1.0/N)*np.matmul(np.transpose(Xtimestheta-data_y),Xtimestheta-data_y)
		print("Loss: \n", Loss)

if __name__ == "__main__":
    main()
