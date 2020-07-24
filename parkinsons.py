# import all necessary libraries
import pandas
import numpy as np
# cross validation is in the lower versions of the library
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt

# load the dataset (local path)
url = "C:/Users/User/Downloads/edu6\Machine learning/Parkinson's Disease/data.csv"
# feature names
features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
            "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE", "status"]
dataset = pandas.read_csv(url, names=features)
# store the dataset as an array for easier processing
array = dataset.values
# X stores feature values
X = array[:, 0:22]
# Y stores the target
Y = array[:, 22]


validation_size = 0.3
# randomize which part of the data is training and which part is validation
seed = 7
# split dataset into training set (70%) and validation set (30%)
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=validation_size, random_state=seed)

# 10-fold cross validation to estimate accuracy (split data into 10 parts; use 9 parts to train and 1 for test)
num_folds = 10
num_instances = len(X_train)
seed = 7
# use the 'accuracy' metric to evaluate models (correct / total)
scoring = 'accuracy'
# algorithms / models
models = []
models.append(('K-Nearest Neighbour', KNeighborsClassifier()))
models.append(('Decision Tree Classifier', DecisionTreeClassifier()))
models.append(('Naive Bayes ', GaussianNB()))

# evaluate each algorithm / model
names = []
score = [5]
i = 0
print("Scores for each algorithm:\n\n")
for name, model in models:
    kfold = RepeatedKFold(n_splits=num_instances,
                          n_repeats=num_folds, random_state=seed)
    cv_results = cross_val_score(
        model, X_train, Y_train, cv=kfold, scoring=scoring)
    names.append(name)

    cls = model.fit(X_train, Y_train)
    print("Name\t\t\t|\t  Accuracy\n-----------------\t|\t------------")
    predictions = model.predict(X_validation)
    print(name, "\t|\t", accuracy_score(Y_validation, predictions)*100)
    print("\n\n")
    print(classification_report(Y_validation, predictions))
    print("\n")
    matrix = confusion_matrix(Y_validation, predictions)
    print("Confusion Matrix:\n", matrix)
    disp = plot_confusion_matrix(cls, X_train, Y_train,
                                 cmap=plt.cm.Blues
                                 )
    disp.ax_.set_title(name)
    print("Normalization Matrix", name, "\n")
    print(disp.confusion_matrix)
    print("\n\n\n")

plt.show()
