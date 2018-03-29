#Author: Usama Munir Sheikh
#The following code implements gradient descent
	#for linear regression using least squares
	#outputs weights
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
	one_vector = np.ones((N,1))
	data_x = np.append(data_x,one_vector,axis=1)
	data_y = data[0:N,-1:];

	#Make Tensor Flow Variables and Placeholders
	X = tf.placeholder("float32",[N,M])
	Y = tf.placeholder("float32",[N,1])
	W = tf.Variable(0.01*tf.ones([M,1]))
	eta = tf.constant(0.001) #learningrate
	LMSE_grad = tf.Variable(1000*tf.ones([M,1]))

	#Write Tensorflow equations and models
	Y_est = tf.matmul(X,W)
	New_LMSE_grad = tf.multiply(tf.divide(1,N), tf.matmul(X, tf.subtract(Y_est,Y),transpose_a=True))
	update_LMSE_grad = tf.assign(LMSE_grad, New_LMSE_grad)
	New_W = tf.subtract(W,tf.multiply(eta,LMSE_grad))
	update_W = tf.assign(W, New_W)

	Loss = tf.multiply(tf.divide(1,N), tf.matmul(tf.subtract(Y_est,Y), tf.subtract(Y_est,Y),transpose_a=True))

	##Initialize model and Compute 
	itr = 1000
	#itr = 1 #uncomment for checking difference in loss as stopping criteria
	error = []
	#prevlossy  = 0.0 #uncomment for checking difference in loss as stopping criteria
	#lossy = 10.0 #uncomment for checking difference in loss as stopping criteria
	#threshold = 0.0001 #uncomment for checking difference in loss as stopping criteria
	model = tf.global_variables_initializer()	
	with tf.Session() as session:
		session.run(model)
		for i in range(itr):
		#while True: #uncomment for checking difference in loss as stopping criteria
			session.run(update_LMSE_grad,feed_dict={X: data_x, Y: data_y})
			lossy = session.run(Loss, feed_dict={X: data_x, Y: data_y})
			result = session.run(update_W)
			error.append(lossy)
		#	if (np.abs(lossy-prevlossy)<threshold):#uncomment for checking difference in loss as stopping criteria
		#		break#uncomment for checking difference in loss as stopping criteria
		#	prevlossy = lossy#uncomment for checking difference in loss as stopping criteria
		#	itr = itr+1#uncomment for checking difference in loss as stopping criteria
		#print(itr) #uncomment for checking difference in loss as stopping criteria
		print("LMS Error Final Value: \n", lossy)
		t = np.arange(itr)
		print("Theta: \n", result)
		plt.plot(t,np.reshape(error,(itr,1)))
		plt.title(r'Fixed-Step Gradient Descent - $\eta$: 0.001')
		plt.xlabel('Number of Iterations')
		plt.ylabel('Value of Loss Function')
		plt.show()
	session.close()	

if __name__ == "__main__":
    main()
