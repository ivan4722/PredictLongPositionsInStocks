# Stock Prediction Model

## Overview

This project uses a machine learning model to predict whether a stock should be long or not based on historical price data. The model uses a sliding window approach to prevent leakage and ensure that the predictions are based on past data only.

## Data Collection

* The data is collected using the `yfinance` library, which provides historical stock price data.
* The data is collected for the ticker symbol "BAC" from January 1, 2020, to August 15, 2024.
* The data is collected at a daily interval.

## Data Preprocessing

* The data is preprocessed to calculate returns and create a target variable indicating whether the stock should be long or not.
* The returns are calculated as the percentage change in the close price over a period of 5 days.
* The target variable is created by thresholding the returns at 0.02.

## Model Training

* The model used is a Random Forest Classifier, which is trained using the `GridSearchCV` class from scikit-learn.
* The model is trained on a sliding window of data, with a window size of 20 days.
* The model is trained with the following hyperparameters:
	+ `n_estimators`: 50 or 100
	+ `max_depth`: 5 or 10
	+ `min_samples_split`: 2 or 5

## Cross Validation

* Cross validation is used to evaluate the model's performance using a stratified k-fold approach.
* The data is split into training and testing sets using a stratified k-fold split with 2 folds.

## Preventing Look-Ahead Bias and Leakage

* To prevent look-ahead bias and leakage, the model is trained on a sliding window of data, and the predictions are made on the next window of data.
* The sliding window approach ensures that the model is only using past data to make predictions, and not future data.
* The data is also split into training and testing sets using a stratified k-fold split with 2 folds, to further prevent leakage.

## Sliding Window Approach

* The sliding window approach is used to make predictions on a rolling basis.
* The model is trained on a window of data, and then the predictions are made on the next window of data.
* The window size is 20 days.

## Plotting Data

* The predicted probabilities are plotted against the actual stock prices to visualize the model's performance.
* The plot shows the close price of the stock over time, with the predicted probabilities represented as a color map.

## Computing Total Model Accuracy

* The total model accuracy is computed by comparing the predicted probabilities with the actual target variable.
* The accuracy is calculated as the percentage of correct predictions out of total predictions.

## Test Accuracy

* The test accuracy is either 0 or 1 because the model is predicting a binary target variable (long or not long).
* A test accuracy of 1 indicates that the model made a correct prediction, while a test accuracy of 0 indicates an incorrect prediction.

## Results

```
Window 220: Train Accuracy = 0.85, Test Accuracy = 0.00
Window 240: Train Accuracy = 0.60, Test Accuracy = 1.00
.
.
.
Window 1100: Train Accuracy = 0.90, Test Accuracy = 1.00
Window 1120: Train Accuracy = 0.60, Test Accuracy = 1.00
Model accuracy over all predictions: 0.74
```
Overall, the model made the correct prediction 74% of the time (whether to long or not)