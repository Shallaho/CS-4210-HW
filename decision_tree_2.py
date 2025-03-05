# -------------------------------------------------------------------------
# AUTHOR: Anthony Codina
# FILENAME: Decision Tree 2
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []
    Total_Acc = []

    # Setting Variables for Accuracy
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # Transform the original categorical training features to numbers and add to the 4D array X.
    # For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    for row in dbTraining:
        x = []
        for i in row:
            if i in ['Young', 'Myope', 'Yes', 'Normal']:
                x.append(1)
            elif i in ['Prepresbyopic', 'Hypermetrope', 'No', 'Reduced']:
                x.append(2)
            elif i == 'Presbyopic':
                x.append(3)
        del x[4]
        X.append(x)

    # Transform the original categorical training classes to numbers and add to the vector Y.
    # For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    for row in dbTraining:
        y = []
        for i in row:
            if i == 'Yes':
                y.append(1)
            if i == 'No':
                y.append(2)
        del y[0]
        Y.append(y)

    # Loop your training and test tasks 10 times here
    for i in range(10):

        # Fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)

        # Read the test data and add this data to dbTest
        dbTest = []


        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:  # skipping the header
                    dbTest.append(row)

        for data in dbTest:
            # Transform the features of the test instances to numbers following the same strategy done during training,
            # and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            # where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            # --> add your Python code here
            xtest = []
            for j in data:
                if j in ['Young', 'Myope', 'Yes', 'Normal']:
                    xtest.append(1)
                elif j in ['Prepresbyopic', 'Hypermetrope', 'No', 'Reduced']:
                    xtest.append(2)
                elif j == 'Presbyopic':
                    xtest.append(3)
            del xtest[4]

            # Transform the original categorical training classes to numbers and add to the vector Y.
            # For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
            ytest = []
            for j in data:
                if j == 'Yes':
                    ytest.append(1)
                if j == 'No':
                    ytest.append(2)
            del ytest[0]
            class_predicted = clf.predict([xtest])[0]
            # Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            if ytest[0] == 1 and class_predicted == 1:
                TP += 1
            if ytest[0] == 1 and class_predicted == 2:
                FP += 1
            if ytest[0] == 2 and class_predicted == 2:
                TN += 1
            if ytest[0] == 2 and class_predicted == 1:
                FN += 1
    # Find the average of this model during the 10 runs (training and test set)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Print the average accuracy of this model during the 10 runs (training and test set).
    # Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f'final accuracy when training on {ds}: {Accuracy}')
