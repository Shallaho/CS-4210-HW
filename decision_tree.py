# -------------------------------------------------------------------------
# AUTHOR: Anthony Codina
# FILENAME: decision_tree.py
# SPECIFICATION: Code for designing a decision tree for contact lenses for CS 4210
# FOR: CS 4210- Assignment #1
# TIME SPENT: 1 hour
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv

db = []
X = []
Y = []

# reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)
            print(row)

# transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
for row in db:
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
print(X)

# transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2
for row in db:
    y=[]
    for i in row:
        if i=='Yes':
            y.append(1)
        if i=='No':
            y.append(2)
    del y[0]
    Y.append(y)
print(Y)

# fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)

# plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes', 'No'], filled=True,
               rounded=True)
plt.show()
