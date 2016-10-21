'''
HackMIT
Preliminary Neural Network
Devin Cody
devin.cody@yale.edu
'''

import numpy as np
import math as m
import matplotlib.pyplot as plt

def sig(x):
	ans = 1/(m.exp(-x) + 1)
	return ans

def twist(x):
	return np.expand_dims(x, axis = 1)

loops = 10
layer_structure = [3,5,7,6,5,4,3,2,1]
eta = .5

### Initialize ANN ###
#Build random weight matrix
def init_weight():
	global weights
	weights = []
	for i in range(len(layer_structure)-1):
		weights.append(np.random.uniform(-1,1,(layer_structure[i+1], layer_structure[i]))) #generated a uniform distribution of weights
	#print "weights:", weights
#Init Input Vector
# init_vect = np.zeros((layer_structure[0],1)) # create a vertical vector of prelimary inputs
# for i in range(layer_structure[0]):
# 	init_vect[i][0] = i # add data to the initial input vector
# print "init_vect", init_vect

### Propagate Forward ###

def propagate_forward(init_vect, weights):
	#Initialize Node Values
	#print "hello"
	inst_values = []
	Lin_comb = np.dot(weights[0],init_vect)
	Sig_bad_format = np.fromiter((sig(e) for e in Lin_comb),float)
	inst_values.append(twist(Sig_bad_format)) #generate first intermediate layer of node values
	for j in range(1,len(layer_structure)-1):
		del Lin_comb
		del Sig_bad_format
		Lin_comb = np.dot(weights[j],inst_values[j-1])
		Sig_bad_format = np.fromiter((sig(e) for e in Lin_comb),float) #generate all other layer node values
		inst_values.append(twist(Sig_bad_format))
	return inst_values

### Propagate Backward ###
# Uses Error terms to update weightings

def propagate_backwards(inst_values, weights, desired, init_vect):
	out = inst_values[-1]
	delta_weight = out*(np.ones((len(out),1)) - out)*(desired-out)

	for i in range(len(layer_structure)-2,0,-1):#[4, 3, 2, 1]
		temp = inst_values[i-1]
		weights[i] = weights[i] + eta*delta_weight*np.transpose(temp)
		delta_weight = temp * (np.ones((len(temp),1)) - temp)*np.dot(np.transpose(weights[i]), delta_weight)
	weights[0] = weights[0] + eta*delta_weight*np.transpose(init_vect)
	return weights

def main():
	data = np.random.uniform(0,1,(loops,3))
	init_weight()
	first= np.zeros(loops)
	for i in range(loops):
		init_vect = twist(data[i])
		output_value = sig(data[i][0]**5 * data[i][1]**1 * data[i][2]**2)# + np.random.uniform(-1,1))
		global weights
		nodes = propagate_forward(init_vect,weights) #Propigate initial values through using weights
		#print "inst_vals", nodes 
		diff = np.copy(nodes)
		diff[-1] = output_value - diff[-1]
		weights = propagate_backwards(nodes, weights, output_value, init_vect) #Update Weights
		first[i] = diff[-1]
	x1 = range(loops)#np.linspace(-10,10,1000)
	#y = np.zeros(1000)
	#for i in range(1000):
	#	y[i] = logistic(x[i])
	plt.figure()
	plt.scatter(x1,first)
	plt.show()

main()







