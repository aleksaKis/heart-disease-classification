#!/usr/bin/env python3

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import graphviz

df = pd.read_csv('./datasets/heart_disease_dataset.csv')

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal']

X = df[features].values
X1 = df[['age', 'cp', 'ca', 'thal']].values
y = df['target']
kf = KFold(n_splits=7, shuffle=True)


# Get more valid scores of model after cross validation
def score_model(x, y, kf, model):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

    print(
        '\nAccuracy score:', np.mean(accuracy_scores),
        '\nPrecision score', np.mean(precision_scores),
        '\nRecall score', np.mean(recall_scores),
        '\nF1 score', np.mean(f1_scores))

    return [np.mean(accuracy_scores), np.mean(precision_scores),
            np.mean(recall_scores), np.mean(f1_scores)]


def display():
    score_model(X, y, kf, model_lg)
    score_model(X, y, kf, dt)

    score_model(X1, y, kf, model_lg)
    score_model(X1, y, kf, dt_final)


# plot_roc_curve(model_lg, X, y)
# plt.savefig('./images/roc_curve.png')
# plt.show()


# Build LogisticRegression model
model_lg = LogisticRegression(solver='liblinear')
model_lg.fit(X, y)
y_lg_pred = model_lg.predict(X)

# Build DecisionTree
dt = DecisionTreeClassifier()
dt.fit(X, y)
y_dt_pred = dt.predict(X)

param_grid = {
    'max_depth': [5, 10, 15, 25],
    'min_samples_leaf': [1, 3, 6],
    'max_leaf_nodes': [5, 15, 30, 50]
}

gs = GridSearchCV(dt, param_grid, scoring='f1', cv=7)
gs.fit(X, y)
# print('Best param_grid: ',gs.best_params_,'\nBest score: ',gs.best_score_)

dt_final = DecisionTreeClassifier(max_depth=5, min_samples_leaf=6, max_leaf_nodes=30)
dt_final.fit(X1, y)

# render dt
dot = export_graphviz(dt_final, feature_names=['age', 'CP', 'CA', 'thal'])


# graph = graphviz.Source(dot)
# graph.render(filename = 'tree', format='png', cleanup=True)

# display()


# Scores for export
def scores_for_table():
    return score_model(X, y, kf, model_lg), score_model(X1, y, kf, dt_final), confusion_matrix(y, y_lg_pred)

display()