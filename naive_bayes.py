# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to
# work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

db = []
X = []
Y = []
# Reading the training data in a csv file
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            db.append(row)

# Transform the original training features to numbers and add them to the 4D array X.
# For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
for data in db:
    x=[]
    for feat in data:
        if feat in ['Sunny','Cool','Normal','Strong']:
            x.append(1)
        elif feat in['Overcast','Mild','High', 'Weak']:
            x.append(2)
        elif feat in['Rain','Hot']:
            x.append(3)
    X.append(x)

# Transform the original training classes to numbers and add them to the vector Y.
# For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    for cla in data:
        if cla=='Yes':
            Y.append(1)
        if cla=='No':
            Y.append(2)

# Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

dbTest = []
Xtest = []
Ytest = []

# Reading the test data in a csv file
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i == 0:
            row.append('Confidence')
            print(row)
        if i > 0:
            dbTest.append(row)

for data in dbTest:
    xtest=[]
    for feat in data:
        if feat in ['Sunny','Cool','Normal','Strong']:
            xtest.append(1)
        elif feat in['Overcast','Mild','High', 'Weak']:
            xtest.append(2)
        elif feat in['Rain','Hot']:
            xtest.append(3)
    Xtest.append(xtest)

# Transform the original training classes to numbers and add them to the vector Y.
# For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    for cla in data:
        if cla=='Yes':
            Ytest.append(1)
        if cla=='No':
            Ytest.append(2)

# Printing the header os the solution

# Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
    pred=clf.predict_proba([xtest])[0]
    if pred[0]>pred[1]:
        if pred[0]>=.75:
            del data[5]
            data.append('Yes')
            data.append(round(float(pred[0]),3))
            print(data)
    else:
        if pred[1]>=.75:
            del data[5]
            data.append('No')
            data.append(round(float(pred[1]),3))
            print(data)
