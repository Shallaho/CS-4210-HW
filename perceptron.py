# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier  # pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

algorithms = ['Perceptron', 'MLPClassifier']
lr=float(input('Enter Learning Rate: '))
shuffle=bool(input('Enter Shuffle: '))
Phi_accuracy = 0
MLPhi_accuracy = 0

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None)  # reading the data by using Pandas library

X_training = np.array(df.values)[:, :64]  # getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:, -1]  # getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None)  # reading the data by using Pandas library

X_test = np.array(df.values)[:, :64]  # getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:, -1]  # getting the last field to form the class label for test

for i in n:  # iterates over n
    for bool in r:  # iterates over r
        for algo in algorithms:  # iterates over the algorithms
            if algo == 'Perceptron':
                clf = Perceptron(eta0=lr, shuffle=shuffle, max_iter=1000)
                # use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init=lr,
                                    hidden_layer_sizes=25, shuffle=shuffle, max_iter=1000)
                # use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
            # hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
            # shuffle = shuffle the training data, max_iter=1000
            # -->add your Pyhton code here

            # Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            # make the classifier prediction for each test sample and start computing its accuracy
            # hint: to iterate over two collections simultaneously with zip() Example:
            # for (x_testSample, y_testSample) in zip(X_test, y_test):
            # to make a prediction do: clf.predict([x_testSample])
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                clf.predict([x_testSample])

            # check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            # and print it together with the network hyperparameters
            # Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            # Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            if algo=='Perceptron':
                P_accuracy=clf.score(X_test,y_test)
                if P_accuracy>Phi_accuracy:
                    Phi_accuracy=P_accuracy
                    print(f'Highest {algo} accuracy so far: {P_accuracy} Parameters: learning rate={lr}, shuffle={shuffle}')
            else:
                MLP_accuracy=clf.score(X_test,y_test)
                if MLP_accuracy>MLPhi_accuracy:
                    MLPhi_accuracy=MLP_accuracy
                    print(f'Highest {algo} accuracy so far: {MLP_accuracy} Parameters: learning rate={lr}, shuffle={shuffle}')
