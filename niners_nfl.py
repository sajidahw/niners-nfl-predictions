"""Case study using NFL data to predict outcome of a football game using the
    Niners offensive and defensive stats from a game during 2022"""

# set up using libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
plt.show()

# LOADING DATA SET
# load dataset
nfl = pd.read_csv('/Users/sajidahwahdy/PycharmProjects/coding_projects/niners-nfl-predictions/nfl/season_2021.csv')

# inspect first few rows
nfl.head()

# summarize outcomes
# check result value counts
nfl.result.value_counts()

# ENCODE RESULT LABELS
# nested dictionary to encode alphanumeric values to numeric values
result_encoder = {'result': {'W': 1, 'T': 0, 'L': 0}}

# encode result column using encoder
nfl.replace(result_encoder, inplace=True)#modified directly inlieu of copy

# check result value counts
nfl['result'].value_counts() # key is column header


# VISUALIZING STATS
# """
# First down conversions by the offense are typically between 12 and 33 in winning games 
# (as depicted by the T-shaped ends of the plot).

# The middle 50% of winning games appears to cover about 20 to 26 first down conversions
# (as depicted by the orange box).

# The middle line indicates a median of about 23 first down conversions by the winning team.

# Summary: winning team typically has a higher number of first downs in a game.
# """

# change stat to view plot
stat = '1stD_offense'

# box plot of stat
stat_plot = sns.boxplot(x='result', y=stat, data=nfl)

# plot labels
stat_plot.set_xticklabels(['loss/tie','win'])
plt.show()
# list feature names
print(nfl.columns[8:])


# DATA PREP
# standardize features
# """
# The functions transformed our stats by subtracting the mean and dividing by the standard deviation. 
# The result is that each stat now has a mean of 0 and a standard deviation of 1. 
# Standardizing benefits: stats in same units and to improve tuning technique
# for better prediction model accuracy.
# """

# select feature variables to be scaled
features = nfl.iloc[:,8:] # saved game stats
scaler = StandardScaler() # function

# fit the transformer to the features
scaler.fit(features) # fx call

# transform and save as X
X = scaler.transform(features) # standardize game stats

# Saving game outcomes
# save result variable as y
y = nfl['result']

# SPLIT TRAINING AND TESTING DATA
# create train-test split of the data for random row selections
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# game stat: X_train, game outcome: y_train
# test size is half of data; random_state is same timeframe for duplication of result check

# RUNNING THE MODEL
# create the classifier
lrc = LogisticRegression()

# fit classifier to the training data
lrc.fit(X_train, y_train)

# CHECKING MODEL ACCURACY
# predict with the classifier using the .predict() function
y_pred = lrc.predict(X_test)

# view the model accuracy with the accuracy_score() function in %
accuracy_score(y_test, y_pred)

# OPTIMIZE BY TUNING HYPERPARAMETERS
# create a list of penalties
penalties = ['l1', 'l2']
# create a list of values for C
C = [0.01, 0.1, 1.0, 10.0, 1000.0]

# logistic regression on data and gets accuracy score for each penalty & C combo
for penalty in penalties:# regularization penalty for too many variables
    for c in C:# inverse of regularization strength to reduce overfitting

        # instantiate the classifier
        lrc_tuned = LogisticRegression(penalty=penalty, C=c, solver='liblinear')

        # fit the classifier to the training data
        lrc_tuned.fit(X_train, y_train)
        
        # predict with the classifier using the .predict() function
        y_pred = lrc_tuned.predict(X_test)

        # view the model accuracy with the accuracy_score() function
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_rd = round(accuracy*100,1)
        
        # print accuracy for each combination of penalty and C
        print(f'Accuracy: {accuracy_rd}% | penalty = {penalty}, C = {c}')


# OPTIMIZE BY CHANGING TEST SIZE
# tuning by changing test size of train-test split for better prediction
# optimal penalty and C
penalty = 'l1'
C = 0.1

# create a list of test_sizes
test_sizes = [val/100 for val in range(20,36)]

for test_size in test_sizes:

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # instantiate the classifier
    lrc_tts = LogisticRegression(penalty = penalty, C = C, solver='liblinear')

    # fit the classifier to the training data
    lrc_tts.fit(X_train, y_train)

    # predict with the classifier using the .predict() function
    y_pred = lrc_tts.predict(X_test)

    # view the model accuracy with the accuracy_score() function
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_rd = round(accuracy*100,1)
    
    # print accuracy for each combination of penalty and test size
    print(f'Accuracy: {accuracy_rd}% | test size = {test_size}')


# SAVING OPTIMIZED MODEL
# set the test size and hyperparameters
test_size = 0.25
penalty = 'l1'
C = 0.1

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# instantiate the classifier
optLr = LogisticRegression(penalty = penalty, C = C, solver='liblinear')

# fit the classifier to the training data
optLr.fit(X_train, y_train)


# VISUALIZE FEATURE IMPORTANCE
# get feature importance for model prediction wins
importance = abs(optLr.coef_[0])

# visualize feature importance as a bar plot
sns.barplot(x=importance, y=features.columns)

# add labels and titles
plt.suptitle('Feature Importance for Logistic Regression')
plt.xlabel('Score')
plt.ylabel('Stat')
plt.show()


# summarize feature importance
for i,v in enumerate(importance.round(2)):
    print(f'Feature: {features.columns[i]}, Score: {v}')


# MODELING NEW DATA
# list of nfl team names to choose from 
list(nfl.team_name.unique())

# set team abbreviation (in capitals) and year
team = 'San Francisco 49ers'
year = 2022

# use helper function to pull new data from Pro Football Reference site
from helper import get_new_data
new_data = get_new_data(team=team, year=year)

# view head of new data
new_data.head()


# Standardizing stats before running model data for predictions
# select just the game stats
new_X = new_data.loc[:,features.columns]

# standardize using original data's scaling
new_X_sc = scaler.transform(new_X)


# Building predictions from data for accuracy scores
# get new predictions
new_preds = optLr.predict(new_X_sc)

# get actual results and set type to float
new_results = new_data['result'].astype(float)

# get accuracy score for new data
acc_score = accuracy_score(new_results, new_preds)

# Table visualization with accuracy score print
# select only game data
col_names = ['day', 'date', 'result', 'opponent', 'tm_score', 'opp_score']
game_data = new_data.loc[:,col_names]
# create comparison table
comp_table = game_data.assign(predicted = new_preds,
                              actual = new_results.astype(int))


# print title and table
print(f'Predicted Wins vs Actual Wins for {team} in {year}')
comp_table

# print accuracy
print(f'\nCurrent Accuracy Score: ' + str(round(acc_score*100,1)) + '%')
