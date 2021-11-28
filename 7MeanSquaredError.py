import numpy as np
import csv
import pandas as pd

data = {'CRIM':[], 'INDUS':[], 'CHAS':[], 'NOX':[], 'RM':[], 'AGE':[], 'DIS':[], 'RAD':[], 'TAX':[], 'PTRATIO':[], 'LSTAT':[], 'MEDV':[]}
trainingData = {'CRIM':[], 'INDUS':[], 'CHAS':[], 'NOX':[], 'RM':[], 'AGE':[], 'DIS':[], 'RAD':[], 'TAX':[], 'PTRATIO':[], 'LSTAT':[], 'MEDV':[]}
testData = {'CRIM':[], 'INDUS':[], 'CHAS':[], 'NOX':[], 'RM':[], 'AGE':[], 'DIS':[], 'RAD':[], 'TAX':[], 'PTRATIO':[], 'LSTAT':[], 'MEDV':[]}

nDims = 12
nData = 506

# Generate Bostion Filter dataset
def GenerateData():
    # Read .csv into dictionary, with each key denoted by a feature (e.g. 'CRIM')
    with open('boston_filter.csv', mode='r') as infile:
        reader = csv.reader(infile)
        header = next(reader)

        if header != None:
            for row in reader:
                data['CRIM'].append(float(row[0]))
                data['INDUS'].append(float(row[2]))
                data['CHAS'].append(float(row[3]))
                data['NOX'].append(float(row[4]))
                data['RM'].append(float(row[5]))
                data['AGE'].append(float(row[6]))
                data['DIS'].append(float(row[7]))
                data['RAD'].append(float(row[8]))
                data['TAX'].append(float(row[9]))
                data['PTRATIO'].append(float(row[10]))
                data['LSTAT'].append(float(row[11]))
                data['MEDV'].append(float(row[12]))

    # Split the data into training / test split
    for key, value in data.items():
        values = value
        # 1/3 of data goes into test set, 2/3 into training
        for i, v in enumerate(values):
            if i % 3 == 0:
                testData[key].append(v)
            else:
                trainingData[key].append(v)
    
    return trainingData, testData


# Extract data for a given feature
def SelectData(trainingData, testData, feature):
    nDataTraining = len(trainingData[feature])
    nDataTest = len(testData[feature])
    X_training = np.zeros((nDataTraining, 1))
    Y_training = trainingData['MEDV']
    X_test = np.zeros((nDataTest, 1))
    Y_test = testData['MEDV']

    for i in trainingData:
        if i == feature:
            X_training = np.asarray(trainingData[i])

    for j in testData:
        if j == feature:
            X_test = np.asarray(testData[j])

    Y_training = np.reshape(Y_training, (nDataTraining, 1))
    Y_test = np.reshape(Y_test, (nDataTest, 1))

    return X_training, Y_training, X_test, Y_test


# Extract data for all features
def AllAttributes(trainingData, testData):
    X_training = np.zeros((len(trainingData['CRIM']), 11))
    X_test = np.zeros((len(testData['CRIM']), 11))

    for i, key in enumerate(data):
        if key != 'MEDV':
            X_training[:, i], Y_training, X_test[:, i], Y_test = SelectData(trainingData, testData, key)
    
    return X_training, Y_training, X_test, Y_test


# Train kernel regressor and return weight vector
def KernelRegressionTrain(X_training, Y_training, gamma, sigma):
    l = len(X_training)
    I_l = np.eye(l)
    
    kernelMatrix = KernelMatrix(X_training, sigma)
    A = kernelMatrix + gamma*l*I_l
 
    alpha = np.linalg.inv(A) @ Y_training

    return alpha

# Predict Y values for test data
def KernelRegressionPredict(X_training, X_test, alpha, sigma):
    l = len(X_training)
    n = len(X_test)
    Y = []

    for i in range(n):
        y_sum = 0
        for j in range(l):
            y = alpha[j]*KernelElement(X_training[j], X_test[i], sigma)
            y_sum += y
        Y.append(float(y_sum))

    return Y


# Calculate kernel function value at point (x1, x2)
def KernelElement(x1, x2, sigma):
        return np.exp(-((np.linalg.norm(x1 - x2)**2)/(2*sigma**2)))


# Generate kernel matrix for all point pairs in X
def KernelMatrix(X, sigma):
    l = len(X)
    n = len(X[0])
    matrix = np.zeros((l, l))

    for i, row1 in enumerate(X):
        for j, row2 in enumerate(X):
            matrix[i, j] = KernelElement(row1, row2, sigma)

    return matrix


# Calculate MSE of predictions
def MSE(X, pred, Y):
    errors = 0

    # Error between prediction and actual Y value
    for i in range(len(X)):
        error = (pred[i] - Y[i])**2
        errors += error
    
    mse = errors / len(X)

    return mse

# The constant function of ones acts as a placeholder for future X data

# Determine values of gamma and sigma that yield the smallest MSE
def OptimalParameters(trainingData, testData):
    runIndex = 0
    Gamma = np.zeros(15)
    g_index = 40
    Sigma = np.zeros(13)
    s_index = 7

    for i in range(len(Gamma)):
        Gamma[i] = 2**(-g_index)
        g_index -= 1

    for j in range(len(Sigma)):
        Sigma[j] = 2**(s_index)
        s_index += 0.5

    # Split data into 5 sets
    trainingData, testData = kFoldCrossValidation(trainingData, testData, k=5)
    meanResults = []
    
    for gamma in Gamma:
        for sigma in Sigma:
            sets = [0, 1, 2, 3, 4]
            crossValidationError = []

            # Use each set as a test set
            for i in range(0, 5):
                sets.remove(i)
                
                # Use remaining sets as training sets
                for j in sets:
                    X_training, Y_training, X_test, Y_test =AllAttributes(trainingData[j], testData[i])
                    alpha = KernelRegressionTrain(X_training, Y_training, gamma, sigma)
                    preds = KernelRegressionPredict(X_training, X_test, alpha, sigma)
                    validationError = MSE(X_test, preds, Y_test)
                    crossValidationError.append(validationError)
                    runIndex += 1

            # Collect mean of cross validation error for each pair of gamma and sigma
            mse = np.mean(crossValidationError)
            meanResults.append((gamma, sigma, mse))

    minResult = min(meanResults, key=lambda x: x[2])
    minGamma = minResult[0]
    minSigma = minResult[1]
    minMSE = minResult[2]

    # Return optimal parameters
    return minGamma, minSigma, minMSE, meanResults


# Split training and test data into 5 random subsets for cross validation
def kFoldCrossValidation(trainingData, testData, k):
    trainingData = pd.DataFrame(trainingData)
    testData = pd.DataFrame(testData)

    # Randomise order of rows in dataset
    shuffledTrainingData = trainingData.sample(frac=1)
    shuffledTestData = testData.sample(frac=1)

    # Split randomised data into 5 subsets
    splitTrainingData = np.array_split(shuffledTrainingData, k)
    splitTestData = np.array_split(shuffledTestData, k)

    for i in range(k):
        splitTrainingData[i] = splitTrainingData[i].to_dict(orient='list')
        splitTestData[i] = splitTestData[i].to_dict(orient='list')

    return splitTrainingData, splitTestData

# Generate best pair of gamma and sigma
trainingData, testData = GenerateData()
X_training, Y_training, X_test, Y_test = AllAttributes(trainingData, testData)
minGamma, minSigma, minMSE, meanResults = OptimalParameters(trainingData, testData)

# Train and test training data using best parameters and print MSE
alpha = KernelRegressionTrain(X_training, Y_training, minGamma, minSigma)
preds = KernelRegressionPredict(X_training, X_training, alpha, minSigma)
mse = MSE(X_training, preds, Y_training)
print(mse)

# Train and test test data using best parameters and print MSE
alpha = KernelRegressionTrain(X_training, Y_training, minGamma, minSigma)
preds = KernelRegressionPredict(X_training, X_test, alpha, minSigma)
mse = MSE(X_test, preds, Y_test)
print(mse)
