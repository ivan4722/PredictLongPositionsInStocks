import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

class RegressionStockLongPosition:
    def __init__(self, ticker, start_date, end_date, prediction_days=5, threshold=0.02):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.prediction_days = prediction_days
        self.threshold = threshold
        self.model = None
        self.scaler = StandardScaler()
        self.data = None

    def get_data(self):
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval='1d')
        self.data['Returns'] = self.data['Close'].pct_change(periods=self.prediction_days).shift(-self.prediction_days)
        self.data['Target'] = (self.data['Returns'] > self.threshold).astype(int)
        self.data.dropna(inplace=True)

    def train_model(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5]}
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42), scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        return best_model

    def rolling_window_prediction(self):
        X = self.data.drop(['Target', 'Returns'], axis=1)
        y = self.data['Target']

        window_size = 20  
        predictions = []
        correct_predictions = 0
        total_predictions = 0

        for i in range(window_size, len(X)):
            X_train = X.iloc[i-window_size:i]
            y_train = y.iloc[i-window_size:i]
            X_test = X.iloc[i:i+1]

            model = self.train_model(X_train, y_train)
            X_test_scaled = self.scaler.transform(X_test)
            probability = model.predict_proba(X_test_scaled)[0]


            if len(probability) == 2:
                probability = probability[1]  
            else:
                probability = probability[0]  

            predictions.append((X_test.index[0], probability))

            if i == len(X) - 1: 
                print(f"Predicted probability of taking a long position on {X_test.index[0].date()}: {probability:.2f}")

        predictions_df = pd.DataFrame(predictions, columns=['Date', 'Probability'])

        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Close'], label='Close Price')
        plt.scatter(predictions_df['Date'], self.data.loc[predictions_df['Date'], 'Close'], c=predictions_df['Probability'], cmap='coolwarm', alpha=0.5)
        plt.colorbar(label='Long Position Probability')
        plt.title(f"{self.ticker} Predicted Probabilities")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()


predictor = RegressionStockLongPosition("AAPL", "2020-01-01", "2024-08-18")
predictor.get_data()
predictor.rolling_window_prediction()
