# This programme further demonstrates the phenomena of overfitting, with a range of basis functions.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


# Returns function values based on theoretical equation
def Function():
    Xf = np.linspace(0, 1, 100)
    F = []

    for x in Xf:
        f = math.sin(2*np.pi*x)
        F.append(f)

    return Xf, F


# Generate y = sin(Apix) for given value of A with uniform error
def Points(A):
    X = []
    P = []

    # Generation of error term
    for i in range(30):
        x = round(np.random.uniform(), 2)
        X.append(x)
    
    for x in X:
        eta = np.random.normal(0, 0.07)
        p = math.sin(A*np.pi*x) + eta
        P.append(p)


    return X, P


# Generates 1000 points around function with uniform error 
def TPoints(a):
    X = []
    P = []

    # Generation of error term
    for i in range(1000):
        x = round(np.random.uniform(), 2)
        X.append(x)
    
    for x in X:
        eta = np.random.normal(0, 0.07)
        p = math.sin(a*np.pi*x)**2 + eta
        P.append(p)

    return X, P


# Polynomial regression function
def FitPolynomial(X, P, K):
    M = len(X)
    phi = np.zeros((M, K))
    
    # Fill feature vectors
    for m in range(M):
        for k in range(K):
            phi[m][k] = X[m]**k

    # Calculate weights
    w = np.dot(np.linalg.inv(np.dot(phi.T, phi)), np.dot(phi.T, P))

    X2 = np.linspace(0, 1, 101)
    N = X2.size

    fit = np.zeros(N)

    # Generate k terms of polynomial equation and add to sum
    for n in range(N):
        eqTotal = 0
        
        for k in range(0, K):
            eqElement = w[k]*(X2[n]**k)
            eqTotal += eqElement

            fit[n] = eqTotal
    
    return X2, fit


# Mean square error function
def MSE(X, X2, P, fit):
    mse = 0

    # Calculate error in each predicted point
    for i, x in enumerate(X):
        index = int(x*100)
        Error = (fit[index] - P[i])**2

        mse += Error

    mse = mse / len(X)
    lnmse = math.log(mse / len(X))

    return lnmse


# Plot function
def Plot(X, F, P):
    ax = plt.subplot(111)
    ax.plot(X, F, color='k')
    ax.scatter(X, P, color='r', s=1.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('x')
    plt.ylabel(r'$\sin^{2}(2πx)$')
    

# Plot single polynomial function
def PlotPolynomial(X, X2, fit, P):
    ax = plt.subplot(111)
    ax.plot(X2, fit, color='r')
    ax.scatter(X, P, color='k', s=1.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('x')
    plt.ylabel(r'$\sin^{2}(2πx)$')
    plt.legend()
    plt.show()
    

# Plot overlap of polynomial functions
def PlotPolynomials(X, X2, fits, P, K):
    colours = iter(cm.rainbow(np.linspace(0, 1, len(fits))))
    ax = plt.subplot(111)
    ax.scatter(X, P, color='k', s=1.5)
    for i, fit in enumerate(fits):
        ax.plot(X2, fit, color=next(colours), label='k=%d' % K[i])  
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('x')
    plt.ylabel(r'$\sin(kπx)$')
    plt.xlim(0, 1)
    plt.ylim(-1.5, 1.5)
    plt.legend(loc='lower center', ncol=3, fontsize = 7)
    plt.show()


# Plot mean square error
def PlotMSE(k, MeanSquaredErrors, a):
    ax = plt.subplot(111)
    ax.scatter(k, MeanSquaredErrors, color='k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('k')
    plt.ylabel('Mean Squard Error')
    plt.show()
    

# Part A(ii): Generate polynomial fits for any k value
def MainA(k):
    X, P = Points(k)
    kValues = []
    fits = []

    for k in range(19):
        X2, fit = FitPolynomial(X, P, k)
        fits.append(fit)
        kValues.append(k)
    
    PlotPolynomials(X, X2, fits, P, kValues)


# Plot MSE against polynomial dimension for any coefficient in sin(Apix) for 30 points
def MainB(a):
    X, P = Points(a)
    lnMSEs = []
    Ks = []

    for k in range(0, 18):
        avlnMSE = 0

        for n in range(100):
            X2, fit = FitPolynomial(X, P, k)
            lnMSE = MSE(X, X2, P, fit)
            avlnMSE += lnMSE

        avlnMSE = avlnMSE / 100
        lnMSEs.append(avlnMSE)
        Ks.append(k)

    PlotMSE(Ks, lnMSEs, a)


# Plot MSE against polynomial dimension for any coefficient in sin(Apix) for 1000 points
def MainC(a):
    X, P = TPoints(a)
    lnMSEs = []
    Ks = []

    for k in range(0, 18):
        avlnMSE = 0

        for n in range(100):
            X2, fit = FitPolynomial(X, P, k)
            lnMSE = MSE(X, X2, P, fit)
            avlnMSE += lnMSE

        avlnMSE = avlnMSE / 100
        lnMSEs.append(avlnMSE)
        Ks.append(k)

    PlotMSE(Ks, lnMSEs, a)

# Generate Desired plot for all values of k from 1 to 18
for i in range(1, 19):
    print(i)
    MainC(i)
