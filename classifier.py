import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

url = "data.csv"
names = [ 'Time Difference Value', 'Skew Rate','Number of retweets','Bot or Not Score','Decision' ]
dataset = pandas.read_csv(url, names=names)
array = dataset.values
X = array[:90,0:4]
Y = array[:90,4]
test_idx = array[90:100,0:4]

clf = DecisionTreeClassifier()
clf.fit(X,Y)

print clf.predict(test_idx)

