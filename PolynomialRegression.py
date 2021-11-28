import numpy as np
import matplotlib.pyplot as plt

# Given Data
x = [1, 2, 3, 4]
y = [3, 2, 0, 5]

M = len(x)
K = 4

# Initialise and fill feature vectors
phi1 = np.zeros((4, 1))
phi2 = np.zeros((4, 2))
phi3 = np.zeros((4, 3))
phi4 = np.zeros((4, 4))

for m in range(M):
    for k in range(K-3):
        phi1[m][k] = x[m]**k

    for k in range(K-2):
        phi2[m][k] = x[m]**k

    for k in range(K-1):
        phi3[m][k] = x[m]**k

    for k in range(K):
        phi4[m][k] = x[m]**k

# Caluclate weights
w1 = np.dot(np.linalg.inv(np.dot(phi1.T, phi1)), np.dot(phi1.T, y))
w2 = np.dot(np.linalg.inv(np.dot(phi2.T, phi2)), np.dot(phi2.T, y))
w3 = np.dot(np.linalg.inv(np.dot(phi3.T, phi3)), np.dot(phi3.T, y))
w4 = np.dot(np.linalg.inv(np.dot(phi4.T, phi4)), np.dot(phi4.T, y))

# Generate space for predictions to be plotted on
X2 =np.arange(0,5,step=0.005)
num_data = X2.size

flat = np.zeros(1000)
linear = np.zeros(1000)
quadratic = np.zeros(1000)
cubic = np.zeros(1000)

# Calculate predicted value at each point using polynomial regression
for i in range(num_data):
    flat[i] = w1[0]
    linear[i] = w2[0] + w2[1]*X2[i]
    quadratic[i] = w3[0] + w3[1]*X2[i] + w3[2]*(X2[i]**2)
    cubic[i] = w4[0] + w4[1]*X2[i] + w4[2]*(X2[i]**2) + w4[3]*(X2[i]**3)

flatMSE = 0
linearMSE = 0
quadraticMSE = 0
cubicMSE = 0

# Calculate mean square error for each fit
for n, m in enumerate(x):
    index = list(X2).index(m)
    flatError = abs(flat[index] - y[n])
    linearError = abs(linear[index] - y[n])
    quadraticError = abs(quadratic[index] - y[n])
    cubicError = abs(cubic[index] - y[n])

    flatMSE += flatError
    linearMSE += linearError
    quadraticMSE += quadraticError
    cubicMSE += cubicError

flatMSE = flatMSE / M
linearMSE = linearMSE / M
quadraticMSE = quadraticMSE / M
cubicMSE = cubicMSE / M

# Print equations
print('\n')
print('k=1 equation is: {:.2f}'.format(w1[0]))
print('k=2 equation is: {:.2f} + {:.2f}x'.format(w2[0], w2[1]))
print('k=3 equation is: {:.2f} - {:.2f}x + {:.2f}x^2'.format(w3[0], abs(w3[1]), w3[2]))
print('k=4 equation is: {:.2f} + {:.2f}x + {:.2f}x^2 + {:.2f}x^3'.format(w4[0], w4[1], w4[2], w4[3]))
print('\n')
print('k=1 Mean Squared Error is: {:.2f}'.format(flatMSE))
print('k=2 Mean Squared Error is: {:.2f}'.format(linearMSE))
print('k=3 Mean Squared Error is: {:.2f}'.format(quadraticMSE))
print('k=4 Mean Squared Error is: {:.2f}'.format(cubicMSE))
print('\n')

# Plot fits
fig=plt.figure()
plt.plot(x, y, 'k.')
plt.plot(X2, flat, 'b', label='k=1')
plt.plot(X2, linear, 'g', label='k=2')
plt.plot(X2, quadratic, 'c', label='k=3')
plt.plot(X2, cubic, 'm', label='k=4')
plt.xlabel('X')
plt.ylabel('Y')
plt.box('off')
fig.axes[0].set_xlim(0,5)
fig.axes[0].set_ylim(-5,8) 
plt.legend()
plt.show()
