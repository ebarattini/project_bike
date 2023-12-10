#!/usr/bin/env python
# coding: utf-8

# # MAP 536 - Python for Data Science - Predicting Cyclist Traffic in Paris

import pandas as pd
import numpy as np
from datetime import datetime
import holidays
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.base import clone
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor


def save_results_to_csv(data, base_filename):
    """
    Converts the input data to a DataFrame (if not already) and saves it to a CSV file with the 
    current date and time appended to the filename. Automatically prints the filename of the saved CSV file.

    Parameters:
    data: Data to be saved, can be a DataFrame, dictionary, list of lists, or a NumPy array.
    base_filename (str): Base filename without extension.
    """
    # Convert the input data to a DataFrame if it's not already one
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data

    # Get the current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct filename with date, time, and .csv extension
    filename = f"{base_filename}_{current_time}.csv"

    # Save DataFrame to CSV
    df.to_csv(filename, index=False)

    
def save_submission_csv(test_data, predictions, model_name):
    """
    Save model predictions to a CSV file with a formatted filename.

    Args:
        test_data (pd.DataFrame): The test data.
        predictions (pd.Series or np.array): Model predictions for the test data.
        model_name (str): The name of the model.

    Returns:
        None
    """
    # Create a dictionary for storing results with Ids and predictions
    results_dict = {'Id': np.arange(test_data.shape[0]), 'log_bike_count': predictions}

    # Convert the dictionary to a DataFrame
    results_df = pd.DataFrame(results_dict)

    # Format the submission CSV filename with model name, date, and time
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    submission_filename = f"submission_{model_name}_{current_datetime}.csv"

    # Save to CSV
    results_df.to_csv(submission_filename, index=False)
    print(results_df)
    

def preprocess_data(train_file, test_file, final_test_file, weather_file, 
                    lockdown_file, holiday_file, subscribers_file, sncf_file):
    """
    Preprocesses the bike sharing data by combining, cleaning, and merging it with 
    external data sources like weather, holidays, and lockdown information.

    Parameters:
    - train_file: Path to the training data file.
    - test_file: Path to the test data file.
    - final_test_file: Path to the final test data file.
    - weather_file: Path to the weather data file.
    - lockdown_file: Path to the lockdown data file.
    - holiday_file: Path to the holiday data file.
    - subscribers_file: Path to the subscribers data file.
    - sncf_file: Path to the SNCF (French National Railway Company) data file.

    Returns:
    - train_data: Preprocessed training data.
    - test_data: Preprocessed test data.
    """

    # Load main datasets
    train_data = pd.read_parquet(train_file)
    test_data = pd.read_parquet(test_file)
    final_test_data = pd.read_parquet(final_test_file)

    # Combining training and test datasets
    combined_train_test = pd.concat([train_data, test_data], axis=0)
    combined_train_test.dropna(inplace=True)
    train_data = combined_train_test
    test_data = final_test_data

    # Load and preprocess external datasets
    weather_data = pd.read_csv(weather_file)
    lockdown_data = pd.read_csv(lockdown_file)
    holiday_data = pd.read_csv(holiday_file)
    velib_subscribers = pd.read_csv(subscribers_file)
    sncf_passengers_delayed = pd.read_csv(sncf_file)

    # Standardize date columns across all datasets
    standardize_date_column(train_data, test_data, final_test_data, weather_data, 
                            lockdown_data, holiday_data, velib_subscribers, sncf_passengers_delayed)

    # Select relevant columns from weather and lockdown data
    weather_data = weather_data[['date', 'feelslike', 'humidity', 'precip', 'windspeed']]
    lockdown_data = lockdown_data[['date', 'school_closures', 'full_lockdown']]

    # Merge external data with main datasets
    external_datasets = [holiday_data, weather_data, lockdown_data, 
                         velib_subscribers, sncf_passengers_delayed]
    train_data = merge_all_external_data(train_data, external_datasets)
    test_data = merge_all_external_data(test_data, external_datasets)

    # Apply transformations: date encoding and temperature binning
    train_data = _encode_dates(train_data)
    train_data = bin_temperature(train_data)
    test_data = _encode_dates(test_data)
    test_data = bin_temperature(test_data)

    # Remove outliers from training data
    train_data = remove_outliers(train_data, 'log_bike_count')

    # Drop unnecessary columns
    columns_to_drop = ['counter_id', 'counter_installation_date', 'counter_technical_id', 
                       'coordinates', 'site_id', 'site_name', 'latitude', 'longitude', 'bike_count']
    train_data.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    test_data.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # Ensure test data has the same feature set as training data, except the target variable
    for column in train_data.columns:
        if column not in test_data.columns and column != 'log_bike_count':
            test_data[column] = 0

    # Align columns in test data to match training data
    test_data = test_data[train_data.columns.drop('log_bike_count')] 

    return train_data, test_data


def standardize_date_column(*dataframes):
    """
    Standardizes the date column across multiple dataframes.

    Parameters:
    - dataframes: A variable number of pandas dataframes.

    For each dataframe, this function renames 'datetime' column to 'date' (if present)
    and converts 'date' column to a pandas datetime object.
    """
    for df in dataframes:
        # Rename 'datetime' column to 'date' if it exists
        if 'datetime' in df.columns:
            df.rename(columns={'datetime': 'date'}, inplace=True)
        
        # Convert 'date' column to datetime object
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        
def merge_all_external_data(main_data, external_datasets):
    """
    Merges the main dataset with a list of external datasets.

    Parameters:
    - main_data: The primary pandas dataframe.
    - external_datasets: A list of pandas dataframes to be merged with the main dataframe.

    The function merges each external dataset with the main dataset on the 'date' column
    and removes any duplicate 'date' columns post-merge.
    """
    for dataset in external_datasets:
        # Merge main_data with an external dataset
        main_data = _merge_external_data(main_data, dataset, 'date')
        
        # Drop duplicate 'date' column if created during the merge
        main_data.drop(columns=['date_y'], inplace=True, errors='ignore')
    
    return main_data


def _encode_dates(X):
    """
    Encodes date-related features in a dataframe.

    Parameters:
    - X: The pandas dataframe with a 'date' column.

    The function extracts various date components, checks for holidays, and creates
    cyclical features for time-based attributes. It also one-hot encodes some of the
    date components and then drops the original date-related columns.
    """
    X = X.copy()
    fr_holidays = holidays.France()  # Get the holiday calendar for France

    # Extract date components
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    X["week"] = X["date"].dt.isocalendar().week

    # Determine if the date is a French holiday
    X["is_holiday"] = X["date"].apply(lambda d: d in fr_holidays).astype(int)

    # Cosine and sine encodings for hours, months, and weekdays
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 23.0)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 23.0)
    X["weekday_sin"] = np.sin(2 * np.pi * X["weekday"] / 6.0)
    X["weekday_cos"] = np.cos(2 * np.pi * X["weekday"] / 6.0)
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 11.0)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 11.0)

    # Season encoding
    X["season"] = X["month"].apply(lambda m: (m % 12 + 3) // 3)
    # Encode seasons as sine and cosine
    X["season_cos"] = np.cos(2 * np.pi * X["season"] / 3.0)
    X["season_sin"] = np.sin(2 * np.pi * X["season"] / 3.0)

    # Rush hour for weekdays not on holidays
    X["morning_rush"] = ((X["weekday"] < 5) & (X["hour"] >= 7) & (X["hour"] <= 9) & (X["is_holiday"] == 0)).astype(int)
    X["evening_rush"] = ((X["weekday"] < 5) & (X["hour"] >= 16) & (X["hour"] <= 18) & (X["is_holiday"] == 0)).astype(int)

    # One-hot encode year and weekday
    year_dummies = pd.get_dummies(X['year'], prefix='year')

    # Concatenate with original DataFrame
    X = pd.concat([X, year_dummies], axis=1)

    # Drop original date components
    X.drop(columns=['year', 'month', 'day', 'weekday', 'hour', 'week', 'date','season'], inplace=True)
    
    return X


def bin_temperature(df):
    """
    Bins temperature values and one-hot encodes the binned categories.

    Parameters:
    - df: The pandas dataframe with a 'feelslike' column representing temperature.

    The function bins the temperature into categories ('cold', 'cool', 'warm', 'hot')
    and then one-hot encodes these categories. It removes the original temperature column
    after binning.
    """
    bins = [-float('inf'), 10, 20, 25, float('inf')]
    labels = ['cold', 'cool', 'warm', 'hot']
    df['temp_binned'] = pd.cut(df['feelslike'], bins=bins, labels=labels)

    # One-hot encode the binned temperatures
    temp_dummies = pd.get_dummies(df['temp_binned'], prefix='temp')
    df = pd.concat([df, temp_dummies], axis=1)

    # Drop original binned column and 'feelslike'
    df.drop(columns=['temp_binned'], inplace=True)
    return df

        
def _merge_external_data(X, df_ext, on_column):
    """
    Merges two dataframes on a specified column using an "asof" merge.

    Parameters:
    - X: The primary pandas dataframe.
    - df_ext: The external pandas dataframe to merge.
    - on_column: The column name to merge on.

    Returns a merged dataframe while preserving the original order of the primary dataframe.
    """
    X = X.copy()
    X["orig_index"] = np.arange(X.shape[0])
    
    # Perform an "asof" merge, which is useful for time-series data
    X = pd.merge_asof(X.sort_values('date'), df_ext.sort_values('date'), on=on_column)
    
    # Restore the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    
    return X
    

def remove_outliers(df, column, multiplier=1.5):
    """
    Remove outliers from a dataframe with Poisson-distributed data by applying
    a square root transformation and using the IQR method.

    :param df: DataFrame to process.
    :param column: The name of the column to check for outliers.
    :param multiplier: The multiplier for the IQR to define what is considered an outlier.
    :return: DataFrame with outliers removed.
    """
    # Apply square root transformation
    transformed_col = np.sqrt(df[column])

    # Compute IQR on the transformed data
    Q1 = transformed_col.quantile(0.10)
    Q3 = transformed_col.quantile(0.90)
    IQR = Q3 - Q1

    # Calculate bounds
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    # Identify outliers and filter them out
    outlier_mask = (transformed_col >= lower_bound) & (transformed_col <= upper_bound)
    return df[outlier_mask]


def main():
    # Loading the data
    train_data_processed, test_data_processed = preprocess_data(
        "train.parquet", "test.parquet", "final_test.parquet",
        "hourly-weather-data.csv", "lockdown-data.csv",
        "paris_school_holidays_2020_2022_correct.csv",
        "velib_subscribers_2020_2022.csv",
        "sncf_passengers_delayed_hourly_2020_2022.csv"
    )

    # Splitting the X and y from the training data
    X = train_data_processed.drop('log_bike_count', axis=1)
    y = train_data_processed['log_bike_count']
    X_test = test_data_processed

    # Define categorical and numerical features
    categorical_features = ['counter_name']
    numerical_features = [
        "feelslike", "humidity", "precip", "windspeed", "Subscribers",
        "SNCFpassengersDelayedInParisPerHour"
    ]

    # Preprocessor for XGBoost
    xgb_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # XGBoost model parameters
    xgb_params = {
        'colsample_bytree': 0.977,
        'gamma': 0.472,
        'learning_rate': 0.118,
        'max_depth': 10,
        'min_child_weight': 4.0,
        'n_estimators': 600,
        'reg_alpha': 0.031,
        'reg_lambda': 0.037,
        'subsample': 0.681,
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'verbosity': 0
    }

    # Initialize XGBoost model
    xgb_model = XGBRegressor(**xgb_params)

    # Create XGBoost pipeline
    xgb_pipeline = Pipeline([
        ('preprocessor', xgb_preprocessor),
        ('model', xgb_model)
    ])

    # Scale 'Subscribers' column in both training and test sets
    X['Subscribers'] /= 400000
    X_test['Subscribers'] /= 400000

    # CatBoost model parameters
    catboost_params = {
        'depth': 12,
        'iterations': 1500,
        'l2_leaf_reg': 3.877,
        'rsm': 0.496,
        'subsample': 0.445,
        'cat_features': [X.columns.get_loc('counter_name')],
        'verbose': 100
    }

    # Initialize CatBoost model
    catboost_model = CatBoostRegressor(**catboost_params)

    # Define StackingRegressor
    stacked_model = StackingRegressor(
        estimators=[
            ('xgboost', xgb_pipeline),
            ('catboost', catboost_model)
        ],
        final_estimator=LinearRegression()
    )

    # Fit the stacked model
    stacked_model.fit(X, y)

    # Make predictions
    predictions = stacked_model.predict(X_test)

    # Save the predictions to a CSV file
    save_submission_csv(predictions, "Submission.csv")


if __name__ == "__main__":
    main()