import numpy as np
import math as m
import matplotlib.pyplot as plt

def logistic(x):
	ans = 1/(m.exp(-x) + 1)
	return ans

x = np.linspace(-10,10,1000)
y = np.zeros(1000)
for i in range(1000):
	y[i] = logistic(x[i])

plt.figure()
plt.scatter(x,y)
plt.show()