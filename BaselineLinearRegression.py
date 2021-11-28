import numpy as np
import csv


data = {'CRIM':[], 'ZN':[], 'INDUS':[], 'CHAS':[], 'NOX':[], 'RM':[], 'AGE':[], 'DIS':[], 'RAD':[], 'TAX':[], 'PTRATIO':[], 'LSTAT':[], 'MEDV':[]}
trainingData = {'CRIM':[], 'ZN':[], 'INDUS':[], 'CHAS':[], 'NOX':[], 'RM':[], 'AGE':[], 'DIS':[], 'RAD':[], 'TAX':[], 'PTRATIO':[], 'LSTAT':[], 'MEDV':[]}
testData = {'CRIM':[], 'ZN':[], 'INDUS':[], 'CHAS':[], 'NOX':[], 'RM':[], 'AGE':[], 'DIS':[], 'RAD':[], 'TAX':[], 'PTRATIO':[], 'LSTAT':[], 'MEDV':[]}

nDims = len(data)
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
                data['ZN'].append(float(row[1]))
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
    X_training = np.zeros((nDataTraining, 2))
    X_test = np.zeros((nDataTest, 2))

    # Append X values and biases to training data
    for i in trainingData:
        if i == feature:
            training = np.asarray(trainingData[i])
            bias = np.ones(len(trainingData[i]))
            for i in range(len(X_training)):
                X_training[i] = (training[i], bias[i])
            Y_training = trainingData['MEDV']

    for j in testData:
        if j == feature:
            test = np.asarray(testData[j])
            bias = np.ones(len(testData[i]))
            for i in range(len(X_test)):
                X_test[i] = (test[i], bias[i])
            Y_test = testData['MEDV']

    return X_training, Y_training, X_test, Y_test


# Extract data for all features
def AllAttributes(trainingData, testData):
    X_training = np.zeros((len(trainingData['CRIM']), 24))
    X_test = np.zeros((len(testData['CRIM']), 24))

    # Each X data column has bias associated som must shift indexing using 2i+1
    for i, key in enumerate(data):
        if key != 'MEDV':
            X_training1, Y_training, X_test1, Y_test = SelectData(trainingData, testData, key)
            X_training[:, 2*i] = X_training1[:, 0]
            X_training[:, 2*i+1] = X_training1[:, 1]
            X_test[:, 2*i] = X_test1[:, 0]
            X_test[:, 2*i+1] = X_test1[:, 1]

    return X_training, Y_training, X_test, Y_test


# Train Linear Regressor and return weights
def LinearRegressionTrain(X_training, Y_training):
    W = np.linalg.lstsq(np.dot(X_training.T, X_training), np.dot(X_training.T, Y_training), rcond=None)[0]

    return W

# Predict Y values for test data
def LinearRegressionPredict(X_test, W):
    preds = X_test @ W

    return preds

# Calculate MSE of predictions
def MSE(X, pred, Y):
    errors = 0

    # Error between prediction and actual Y value
    for i, x in enumerate(X):
        error = (pred[i] - Y[i])**2
        errors += error
    
    mse = errors / len(X)

    return mse

# MSE of test data is 92.48861611174934
# MSE of training data is 80.37339221090267

# The constant function of ones acts as a placeholder for future X data


# Print MSE for training and test sets, averaged over 20 runs for each feature and all features
def MeanSquareErrors():
    features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT']
    mseTrain = 0
    mseTest = 0
    
    for i in range(20):
        feature = 'ZN'
        trainingData, testData = GenerateData()
        X_training, Y_training, X_test, Y_test = SelectData(trainingData, testData, feature)
        #X_training, Y_training, X_test, Y_test = AllAttributes(trainingData, testData)
        W = LinearRegressionTrain(X_training, Y_training)
        predsTrain = LinearRegressionPredict(X_training, W)
        mseTrain += MSE(X_training, predsTrain, Y_training)
        predsTest = LinearRegressionPredict(X_test, W)
        mseTest += MSE(X_test, predsTest, Y_test)
    print('Train MSE for {} is {}'.format(feature, np.sqrt(mseTrain/20)))
    print('Test MSE for {} is {}'.format(feature, np.sqrt(mseTest/20)))


# Print SD for training and test sets, averaged over 20 runs for each feature and all features
def StandardDeviations():
    features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT']
    sdTrain = 0
    sdTest = 0

    for i in range(20):
        feature = 'ZN'
        mseTrainAverage = 71
        mseTestAverage = 73.1
        trainingData, testData = GenerateData()
        X_training, Y_training, X_test, Y_test = SelectData(trainingData, testData, feature)
        #X_training, Y_training, X_test, Y_test = AllAttributes(trainingData, testData)
        W = LinearRegressionTrain(X_training, Y_training)
        predsTrain = LinearRegressionPredict(X_training, W)
        mseTrain = MSE(X_training, predsTrain, Y_training)
        sdTrain += abs(mseTrain - mseTrainAverage)**2
        predsTest = LinearRegressionPredict(X_test, W)
        mseTest = MSE(X_test, predsTest, Y_test)
        sdTest += abs(mseTest - mseTestAverage)**2
    print('Train SD for {} is {}'.format(feature, np.sqrt(sdTrain/20)))
    print('Test SD for {} is {}'.format(feature, np.sqrt(sdTest/20)))
