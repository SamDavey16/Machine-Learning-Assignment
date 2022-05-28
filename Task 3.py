from sklearn import neural_network
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

pd.options.display.max_columns = None
pd.options.display.max_rows = None
df = pd.read_csv (r"Task3 - dataset - HIV RVG.csv")
print(df.describe(include='all'))
df.boxplot()
ax = df.plot.kde(bw_method=0.3)
plt.show()

x = df.loc[:,("Alpha", "Beta", "Lambda", "Lambda1", "Lambda2")]
y = df.loc[:,("Participant Condition")]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

epochs = [50, 100, 500, 1000, 10000]
accuracy_ann = []
accuracy_rt = []
for i in epochs:
    mlp = MLPClassifier(hidden_layer_sizes=(8,8), activation='relu', solver='adam', max_iter=i, learning_rate = 'invscaling')
    mlp.fit(x_train,y_train)

    predict_train = mlp.predict(x_train)
    predict_test = mlp.predict(x_test)

    acc = accuracy_score(y_test, predict_test)
    accuracy_ann.append(acc)
    print("Artificial Neural Network: ", acc)

    clf=RandomForestClassifier(n_estimators=i)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    Accuracy = metrics.accuracy_score(y_test, y_pred)
    accuracy_rt.append(Accuracy)
    print("Random Forest: ", Accuracy)

plt.plot(epochs,accuracy_ann)
plt.plot(epochs,accuracy_rt)
plt.show()

df = pd.read_csv (r"Task3 - dataset - HIV RVG.csv")
x = df.loc[:,("Alpha", "Beta", "Lambda", "Lambda1", "Lambda2")]
y = df.loc[:,("Participant Condition")]
folds = range(2,10)
print("Artificial Neural Network 10 fold cv")
for k in folds:
    kf = KFold(n_splits=k)
    scores = cross_val_score(mlp, x, y, scoring='accuracy', cv=kf, n_jobs=-1)
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print("Random trees 10 fold cv")
for k in folds:
    kf = KFold(n_splits=k)
    scores = cross_val_score(clf, x, y, scoring='accuracy', cv=kf, n_jobs=-1)
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
