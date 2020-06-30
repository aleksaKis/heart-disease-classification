#!/usr/bin/env python3

from heart_disease import scores_for_table
import plotly.graph_objects as go
import pandas as pd

df = pd.read_csv('./datasets/heart_disease_dataset.csv')
df = df.sort_values(by=['age'])
scores = scores_for_table()

lr_formated_scores = [round(sc, 5) for sc in scores[0]]
dt_formated_scores = [round(sc, 5) for sc in scores[1]]

features = ['<b>' + feature.capitalize() + '</b>' for feature in list(df.keys())]

# DataFrame
data_header = dict(values=features,
                   align='center',
                   line_color='black',
                   fill_color='lightgray',
                   font=dict(color='black', size=14))

data_cells = dict(values=[df.age, df.sex, df.cp, df.trestbps, df.chol, df.fbs, df.restecg, df.thalach,
                          df.exang, df.oldpeak, df.slope, df.ca, df.thal, df.target],
                  font=dict(color='black', size=14))

data_fig = go.Figure(data=[go.Table(header=data_header, cells=data_cells)])
data_fig.show()

# Scores
scores_header = dict(values=["<b>Accuracy</b>", "<b>Precision</b>",
                             "<b>Recall</b>", "<b>F1</b>"],
                     align='center',
                     font=dict(color='black', size=14))

scores_cells = dict(values=lr_formated_scores, font=dict(color='black', size=14))

scores_fig = go.Figure(data=[go.Table(header=scores_header, cells=scores_cells)])
scores_fig.show()

dt_scores_header = dict(values=["<b>Accuracy</b>", "<b>Precision</b>",
                                "<b>Recall</b>", "<b>F1</b>"],
                        align='center',
                        font=dict(color='black', size=14))

dt_scores_cells = dict(values=dt_formated_scores, font=dict(color='black', size=14))

dt_scores_fig = go.Figure(data=[go.Table(header=dt_scores_header, cells=dt_scores_cells)])
dt_scores_fig.show()

# Confusion Matrix
cm_header = dict(values=["<b>Confusion Matrix</b>", "<b>Actual Positive (1)</b>", "<b>Actual Negative (0)</b>"],
                 align='center', font=dict(color='black', size=14))
cm_cells = dict(values=[["<b>Predicted Positive (1)</b>", "<b>Predicted Negative (0)</b>"], scores[2][0], scores[2][1]],
                align='center', font=dict(color='black', size=14))

cm_fig = go.Figure(data=[go.Table(header=cm_header, cells=cm_cells)])
cm_fig.show()
