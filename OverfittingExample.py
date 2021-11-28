# This programme illustrates the phenomena of overfitting.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


# Returns function values based on theoretical equation
def Function():
    Xf = np.linspace(0, 1, 100)
    F = []

    for x in Xf:
        f = math.sin(2*np.pi*x)**2
        F.append(f)

    return Xf, F


# Generates 30 points around function with uniform error
def Points():
    X = []
    P = []

    # Generation of error term
    for i in range(30):
        x = round(np.random.uniform(), 2)
        X.append(x)
    
    for x in X:
        eta = np.random.normal(0, 0.07)
        p = math.sin(2*np.pi*x)**2 + eta
        P.append(p)

    return X, P

# Generates 1000 points around function with uniform error 
def TPoints():
    X = []
    P = []

    # Generation of error term
    for i in range(1000):
        x = round(np.random.uniform(), 2)
        X.append(x)
    
    for x in X:
        eta = np.random.normal(0, 0.07)
        p = math.sin(2*np.pi*x)**2 + eta
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

    # Plot for each polynomial fit
    for i, fit in enumerate(fits):
        ax.plot(X2, fit, color=next(colours), label='k=%d' % K[i]) 
    ax.scatter(X, P, color='k', s=1.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('x')
    plt.ylabel(r'$\sin^{2}(2πx)$')
    plt.xlim(0, 1)
    plt.ylim(-0.5, 1.5)
    plt.legend()
    plt.show()


# Plot mean square error
def PlotMSE(k, MeanSquaredErrors):
    ax = plt.subplot(111)
    ax.scatter(k, MeanSquaredErrors, color='k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('k')
    plt.ylabel('Mean Squard Error')
    plt.show()
    

# Part A(ii): Generate polynomial fits for given k values
def MainA():
    X, P = Points()
    kValues = [2, 5, 10, 14, 18]
    fits = []

    for k in range(19):
        avfit = np.zeros(101)

        if k in kValues:
            for n in range(100):
                X2, fit = FitPolynomial(X, P, k)
                avfit += fit
        else:
            continue
        avfit = avfit / 100
        fits.append(avfit)
    
    PlotPolynomials(X, X2, fits, P, kValues)


# Plot MSE against polynomial dimension for 30 points
def MainB():
    X, P = Points()
    lnMSEs = []
    Ks = []

    for k in range(0, 18):
        avlnMSE = 0

        for n in range(100):
            X2, fit = FitPolynomial(X, P, k)
            lnMSE = MSE(X, X2, P, fit)
            avlnMSE += lnMSE

        lnMSEs.append(avlnMSE)
        Ks.append(k)

    PlotMSE(Ks, lnMSEs)


# Plot MSE against polynomial dimension for 1000 points
def MainC():
    X, P = TPoints()
    lnMSEs = []
    Ks = []

    for k in range(0, 18):
        avlnMSE = 0

        for n in range(100):
            X2, fit = FitPolynomial(X, P, k)
            lnMSE = MSE(X, X2, P, fit)
            avlnMSE += lnMSE

        lnMSEs.append(avlnMSE)
        Ks.append(k)

    PlotMSE(Ks, lnMSEs)

def MainD():
    avlnMSEs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,]

    for i in range(100):
        lnMSEs = MainC()
        avlnMSEs = [sum(x) for x in zip(lnMSEs, avlnMSEs)]

    PlotMSE(k, avlnMSEs)
