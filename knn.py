# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
X = []
Y= []
incorrect=0
total=0
rows=0

# Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            rows+=1
            db.append(row)

count=0
# Loop your data to allow each instance to be your test set
for run in range(rows):
    X=[]
    Y=[]
    for i, row in enumerate(db):
        if i==count:
            x= []
            for val in range(len(row)-1):
                x.append(float(row[val]))
            testSample=x
            if row[len(row)-1]=='spam':
                Ysample=1
            else:
                Ysample=2
        else:
            # Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
            # For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
            # Convert each feature value to float to avoid warning messages
            feat_val=[]
            for num in range(len(row)-1):
                a=float(row[num])
                feat_val.append(a)
            X.append(feat_val)

            # Transform the original training classes to numbers and add them to the vector Y.
            # Do not forget to remove the instance that will be used for testing in this iteration.
            # For instance, Y = [1, 2, ,...].
            # Convert each feature value to float to avoid warning messages
            if row[len(row)-1]=='spam':
                y=1
                Y.append(float(y))
            else:
                y=2
                Y.append(float(y))


    # Fitting the knn to the data
    count+=1
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    # Use your test sample in this iteration to make the class prediction. For instance:
    # class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    print(testSample)
    class_predicted = clf.predict([testSample])[0]
    print(class_predicted)
    # Compare the prediction with the true label of the test instance to start calculating the error rate.
    if Ysample==class_predicted:
        total+=1
    else:
        incorrect+=1
        total+=1

# Print the error rate
print(incorrect)
print(total)
print(incorrect/total)
