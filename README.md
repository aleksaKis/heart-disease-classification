# Classification - Heart Disease

This is the small project I made while learning and practicing machine learning in programming language python. I tried to get used to and get familiar with things I learned so far before I dig on in more complex things in this branch of data science. In the project I build two final models from LogisticRegression and DecisionTree classes, witch had best scores. While this is supervised learning, using Sci kit learn we simulated unsupervised learning by splitting dataset in test and train data. To get better scoring we use split method with generate indices for test and train set randomly, then we calculate mean value of all scores to get more accurate precision of model.

Project was made by using dataset [Heart Disease UCI](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).


### LogisticRegression 

<img src='https://raw.githubusercontent.com/aleksaKis/Classification-Heart-disease/master/images/lr_scores.png' alt='Logistic Regression scores'>
<i>Score table of LR model using all features.</i>


<img src='https://raw.githubusercontent.com/aleksaKis/Classification-Heart-disease/master/images/lr_scores.png' alt='Confusion Matrix'>
<i>Confusion Matrix table witch shows how many data points were predicted wrong.</i>


<img src='https://raw.githubusercontent.com/aleksaKis/Classification-Heart-disease/master/images/roc_curve.png' alt='Roc curver graph'>
<i>Roc curve is showing performance of many models with each using different treshold</i>

### Decision Tree

<img src='https://raw.githubusercontent.com/aleksaKis/Classification-Heart-disease/master/images/dt_scores.png' alt='Decision Tree scores'>
<i>Score table of DT model using all features.</i>


<img src='https://raw.githubusercontent.com/aleksaKis/Classification-Heart-disease/master/images/Simplified_tree.png' alt='Simplified tree'>
<i>Simplified DT version, if you want to see full tree with all branches follow the <a href='https://github.com/aleksaKis/Classification-Heart-disease/blob/master/images/tree.png'>link</a></i>
