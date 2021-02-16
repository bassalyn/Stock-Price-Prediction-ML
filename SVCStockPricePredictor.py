# Algorithmic Trading Software
# By Bradley Assaly-Nesrallah
#
# Description: Uses the OHLC data to derive two primary features the Relative Strength Index
# and the Moving Average Convergence Divergence indicator for inputs
# The dataset is given a label buy or sell which represents if the stock appreciated in price the following day
# These are given to a Support Vetor Classifier with a polynomial kernel for classification
# The hyperparameters of the SVC are tuned to be more bullish, this represents the increasing trend of the market
# However this does not take into account costs of each trade, and other negative

#import libraries numpy, pandas and sklearn
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC


#import the data of a CSV with OHLC and Volume data
data = pd.read_csv("S&Pdata.csv", index_col=0)

# We engineer features to add to the data, RSI and MACD are the two primary indicators we create

# Create a 1 day momentum feature
data['Momentum_1D'] = data['Close'] - data['Close'].shift()

# Define a function to compute the relative strength indicator for quantitative analysis
# We use the 14 day momentum for this computation since it is common in industry
def rsi(values):
    up = values[values>0].mean()
    down = -1*values[values<0].mean()
    return 100 * up / (up + down)

# Compute the 14 day RSI feature using the one day momentum feature and applying the transform function
data['RSI_14D'] = data['Momentum_1D'].rolling(center=False,window=14).apply(rsi)

# Compute the Moving Average Convergence Divergence indicator by using the 26D and 12D Exponential moving averages
data['EMA26D'] = data['Close'].ewm(span=26).mean()
data['EMA12D'] = data['Close'].ewm(span=12).mean()
data['MACD'] = data['EMA12D'] - data['EMA26D']

# Create and assign labels where 1 is buy and 0 is sell based on if price appreciates the following day
data["Label"] = np.where(data["Close"] > (data["Close"].shift(-1)), 0, 1)

# Remove columns not relevant to the SVC
data = data.drop(['Open','Close','High','Low','EMA26D','EMA12D','Momentum_1D', 'Adj Close', 'Volume'], axis=1)


# Convert index to data time format
data.index = pd.to_datetime(data.index)

# Drop any NA cells
data=data.dropna(how="any")

# Split the data into train and test sets, we use most of the data for training and the more recent for testing
train_data = data['1990-01-01':'2010-12-31'].copy()
test_data = data['2011-01-01':'2018-01-01'].copy()

# function to rebalance the training set
def rebalance(unbalanced_data):
    data_minority = unbalanced_data[unbalanced_data["Label"] == 0]
    data_majority = unbalanced_data[unbalanced_data["Label"] == 1]
    n_samples = len(data_majority)
    data_minority_upsampled = resample(data_minority, replace=True, n_samples=n_samples, random_state=5)
    data_upsampled = pd.concat([data_majority, data_minority_upsampled])
    data_upsampled.sort_index(inplace=True)
    data_upsampled["Label"].value_counts()
    return data_upsampled

# rebalance the training data
train_data = rebalance(train_data)

# Normalize the data
scaler = sk.preprocessing.MinMaxScaler()
scaler.fit(train_data)
train_data.loc[:, train_data.columns] = scaler.transform(train_data)
test_data.loc[:,test_data.columns] = scaler.transform(test_data)

# Split train and test data and assign labels
X_train = train_data.drop("Label",1)
y_train = train_data["Label"]
X_test = test_data.drop("Label",1)
y_test = test_data["Label"]

# Create a classifier using the Support Vector Classifier from Scikit Learn
# The hyperparamets of the SVC are polynomial kernel of degree 5, C value of 0.5, and auto gamma to reflect
# The upward trend of the S&P over time
clf = SVC(kernel="poly", degree=5,gamma="auto", C=0.5)
# Train the model here
clf.fit(X_train, y_train)
# And make predictions here
predictions = clf.predict(X_test)
#compute the accuracy of the predictions
accuracytest = accuracy = (accuracy_score(y_test, predictions))
#display the result for the reader
print("Accuracy on test set was {0}%.".format(accuracytest*100))
print("Here is the confusion matrix for the test data")
print(confusion_matrix(y_test,predictions))