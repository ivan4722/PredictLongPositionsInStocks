# Stock Prediction Model

## Overview

This project implements a machine learning model that predicts whether a stock should be taken as a long position based on historical price data. The model uses a rolling window approach to ensure predictions are based on past data only, preventing data leakage and look-ahead bias.

## Data Collection

- **Source:** The historical stock price data is collected using the `yfinance` library.
- **Ticker:** The data is collected for the ticker symbol "AAPL".
- **Date Range:** The data spans from January 1, 2020, to August 18, 2024.
- **Interval:** The data is collected at a daily interval.

## Data Preprocessing

- The data is preprocessed to calculate returns and create a target variable indicating whether the stock should be taken as a long position.
- Returns are calculated as the percentage change in the close price over a period of 5 days.
- The target variable is created by thresholding the returns at 0.02.

## Model Training

- The model used is a Random Forest Classifier, which is trained using the `GridSearchCV` class from scikit-learn.
- The model is trained on a rolling window of data, with a window size of 20 days.
- The model is trained with the following hyperparameters:
  - `n_estimators`: 50 or 100
  - `max_depth`: 5 or 10
  - `min_samples_split`: 2 or 5

## Cross Validation

- Cross-validation is used to evaluate the model's performance using a stratified k-fold approach.
- The data is split into training and testing sets using a stratified k-fold split with 2 folds.

## Preventing Look-Ahead Bias and Leakage

- To prevent look-ahead bias and leakage, the model is trained on a rolling window of data, and the predictions are made on the next window of data.
- The rolling window approach ensures that the model is only using past data to make predictions, and not future data.
- The data is also split into training and testing sets using a stratified k-fold split with 2 folds, to further prevent leakage.

## Rolling Window Approach

- The rolling window approach is used to make predictions on a continuous basis.
- The model is trained on a window of data, and then the predictions are made on the next window of data.
- The window size is 20 days.

## Plotting Data

- The predicted probabilities are plotted against the actual stock prices to visualize the model's performance.
- The plot shows the close price of the stock over time, with the predicted probabilities represented as a color map.

## Computing Total Model Accuracy

- The total model accuracy is computed by comparing the predicted probabilities with the actual target variable.
- The accuracy is calculated as the percentage of correct predictions out of total predictions.

## Test Accuracy

- The test accuracy is either 0 or 1 because the model is predicting a binary target variable (long or not long).
- A test accuracy of 1 indicates that the model made a correct prediction, while a test accuracy of 0 indicates an incorrect prediction.
- A correct prediction is calculated as follows:
  using ```prediction_days``` and ```threshold```, does the long generate a ```threshold```% profit after ```prediction_days``` days? 

## Results

```
Accuracy: 0.65
Precision: 0.28
Recall: 0.36
F1 Score: 0.32
AUC-ROC Score: 0.55
```
Using prediction days = 5 and threshold = 0.02 (2% profit)
Overall, the model made the correct prediction 65% of the time (whether to long or not) with stock T
It is important to note that T does not have an uptrend on our desired time frame (which would skew results)
![Stock Prediction Model Results](https://ix221-images.s3.us-east-2.amazonaws.com/BACpreds.png)

EDIT:
part of the README may be outdated, I changed the model to use a rolling window to predict possibility of taking a long position on a present day.
EX: ```Predicted probability of taking a long position on 2024-08-09: 0.24```