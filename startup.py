import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data_op6.csv', delimiter=',', dtype = np.float32)

lambda_0 = data[:,0]
lambda_1 = 50
d_1 = 60
lambda_2 = 600
d_2 = 70
lambda_3 = 680
d_3 = 50

s_1 = np.exp(-(lambda_0-lambda_1)**2/d_1**2)
s_2 = np.exp(-(lambda_0-lambda_2)**2/d_2**2)
s_3 = np.exp(-(lambda_0-lambda_3)**2/d_3**2)

s_tot = data[:,1]
plt.figure(1)
plt.plot(lambda_0, s_1)
plt.plot(lambda_0, s_2)
plt.plot(lambda_0, s_3)
plt.plot(lambda_0, s_tot)

A_transpose = np.array([s_1,s_2,s_3])

A_matrix = np.transpose(A_transpose)
A_transpose_multiplied_with_A = A_transpose @ A_matrix

x_vector = np.linalg.solve(A_transpose_multiplied_with_A, A_transpose @  s_tot)
print(x_vector)
plt.plot(lambda_0, np.dot(x_vector, A_transpose))
plt.show()