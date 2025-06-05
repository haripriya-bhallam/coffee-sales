# coffee-sales
# Author: Haripriya Bhallam
# Date: 2025
# Objective: Explore customer behavior and forecast coffee sales using a data science.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Load data
data = pd.read_csv("coffee_sales.csv")

# Step 1: Clean and Prepare the Dataset
def preprocess_data(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df['date'] = pd.to_datetime(df['date'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Create time-based features
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.to_period('M').astype(str)
    
    # Replace missing card values
    df['card'] = df['card'].fillna('CASH_USER')
    
    return df

data = preprocess_data(data)

# Step 2: Feature Engineering
def engineer_features(df):
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['is_morning'] = df['hour'].between(6, 11).astype(int)
    df['is_evening'] = df['hour'].between(17, 22).astype(int)
    df['coffee_category'] = df['coffee_name'].apply(lambda x: 'Milk-Based' if 'Latte' in x or 'Cappuccino' in x else 'Black-Based')
    
    return df

data = engineer_features(data)

# Step 3: Aggregate for Forecasting
sales_data = (
    data.groupby(['date', 'coffee_name'])
    .agg(total_sales=('money', 'sum'),
         transactions=('money', 'count'))
    .reset_index()
)

# Fill in missing dates for consistent modeling
all_dates = pd.date_range(sales_data['date'].min(), sales_data['date'].max())
unique_coffees = sales_data['coffee_name'].unique()

full_index = pd.MultiIndex.from_product([all_dates, unique_coffees], names=['date', 'coffee_name'])
sales_data = sales_data.set_index(['date', 'coffee_name']).reindex(full_index, fill_value=0).reset_index()

# Step 4: Time-Series Modeling
model_data = sales_data.copy()
model_data['day_of_week'] = model_data['date'].dt.dayofweek
model_data['month'] = model_data['date'].dt.month

# Create lag features
for lag in [1, 3, 7]:
    model_data[f'lag_{lag}'] = model_data.groupby('coffee_name')['total_sales'].shift(lag)

model_data = model_data.dropna()

# Train/Test Split using TimeSeriesSplit
features = ['day_of_week', 'month', 'lag_1', 'lag_3', 'lag_7']
X = model_data[features]
y = model_data['total_sales']

tscv = TimeSeriesSplit(n_splits=5)
model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1)

maes = []
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    maes.append(mae)

print("Time Series Cross-Validation MAE:", round(np.mean(maes), 2))

# Step 5: Visualization 
plt.figure(figsize=(12, 6))
sample_product = model_data['coffee_name'].unique()[0]
plot_df = model_data[model_data['coffee_name'] == sample_product].tail(60)

plt.plot(plot_df['date'], plot_df['total_sales'], label="Actual", marker='o')
plt.plot(plot_df['date'], model.predict(plot_df[features]), label="Predicted", linestyle='--', color='orange')
plt.title(f"Sales Forecasting for {sample_product}")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
