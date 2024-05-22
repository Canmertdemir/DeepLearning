import numpy as np
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sktime.performance_metrics.forecasting import mean_squared_error
from DataPreprocessingTools.data_prep_library import rename_columns, variable_fix, feature_importance

# Pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)

# Load the dataset
data = pd.read_csv(r"C:\Users\pc\DeepLearning\DataSet\household_power_consumption.txt", sep=";", header=None,
                   low_memory=False)
rename_columns(data)


def variable_fix(df):
    df = df.drop(0, axis=0)
    df.columns = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                  'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    # Convert Date and Time to datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='%d/%m/%Y')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time

    # Extract Year, Month, Day, Hour, and Minute
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
    df['Minute'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.minute

    # Combine Date and Time into a single datetime column
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df = df.set_index('datetime')

    # Drop unnecessary columns
    df = df.drop(['Date', 'Time', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], axis=1)

    # Convert numeric columns and handle missing values
    numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].fillna(df[column].mean())

    return df
data = variable_fix(data)


def time_series_split(df, target, n_splits):
    X = df.drop(target, axis=1)
    y = df[target]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_indices, test_indices = next(tscv.split(X))

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = time_series_split(df=data, target='Global_active_power', n_splits=5)


def model_stacking(Xtrain, Xtest, ytrain, ytest):
    lgbm_reg = LGBMRegressor(n_estimators=200, learning_rate=0.01, max_depth=8)
    xgb_reg = XGBRegressor(n_estimators=200, eta=0.01, max_depth=6)

    regressors = [('lgbm', lgbm_reg), ('xgb', xgb_reg)]
    stacking_reg = StackingRegressor(estimators=regressors, final_estimator=LGBMRegressor(), n_jobs=-1)
    stacking_reg.fit(Xtrain, ytrain)

    stacking_train_pred = stacking_reg.predict(Xtrain)
    stacking_test_pred = stacking_reg.predict(Xtest)

    stacking_train_mse = mean_squared_error(ytrain, stacking_train_pred)
    stacking_test_mse = mean_squared_error(ytest, stacking_test_pred)

    print('Train RMSE:', np.sqrt(stacking_train_mse))
    print('Test RMSE:', np.sqrt(stacking_test_mse))

    joblib.dump(stacking_reg, 'stacking_model.pkl')

    loaded_model = joblib.load('stacking_model.pkl')
    predictions = loaded_model.predict(Xtest)

    feature_importance(data, X_train, y_train, save_path="feature_importance_plot.png")

    return predictions

predictions = model_stacking(X_train, X_test, y_train, y_test)

# Plotting residuals (errors) between predictions and true values
def plot_residuals(y_true, predictions, save_path=None):
    residuals = y_true - predictions
    plt.figure(figsize=(10, 5))
    plt.plot(y_true.index, residuals, label='Residuals', marker='o', linestyle='none')
    plt.hlines(0, y_true.index.min(), y_true.index.max(), colors='r', linestyles='dashed')
    plt.title('Residual Plot')
    plt.xlabel('DateTime')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()

plot_residuals(y_test, predictions, save_path='residual_plot.png')

# Plotting test vs prediction
def test_vs_prediction(df, y_test, predictions, save_path=None):
    plt.figure(figsize=(10, 5))

    # Get the datetime index from y_test and match it with predictions
    df_dates = y_test.index[:60]

    plt.plot(df_dates, y_test[:60], label='Test Verileri', marker='o')
    plt.plot(df_dates, predictions[:60], label='Tahmin Verileri', marker='x')

    plt.title('Test Verileri ve Tahmin Verileri Karşılaştırması')
    plt.xlabel('Tarih')
    plt.ylabel('Değerler')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()

test_vs_prediction(data, y_test, predictions, save_path='EnsembleML_Model_prediction_vs_Test.png')

