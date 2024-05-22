import numpy as np
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sktime.performance_metrics.forecasting import mean_squared_error
from DataPreprocessingTools.data_prep_library import rename_columns, variable_fix, feature_importance

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)

data = pd.read_csv(r"C:\Users\pc\DeepLearning\DataSet\household_power_consumption.txt", sep=";", header=None, low_memory=False)
def data_processer(df):
    rename_columns(df)
    df = variable_fix(df)
    return df
data=data_processer(data)
def train_test_seperation(df, target, test_size, random):
    x = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random, shuffle=True)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_seperation(df=data, target = "Global_active_power", test_size = 0.3, random=42)

def model_stacking(Xtrain, Xtest, ytrain, ytest):

    lgbm_reg = LGBMRegressor(n_estimators=200, learning_rate=0.01, max_depth=8)
    xgb_reg = XGBRegressor(n_estimators=200, eta=0.01, max_depth=6)

    regressors = [('lgbm', lgbm_reg), ('xgb', xgb_reg)]
    stacking_reg = StackingRegressor(estimators=regressors, final_estimator=LGBMRegressor(), n_jobs=-1)
    stacking_reg.fit(Xtrain, ytrain)

    stacking_train_pred = stacking_reg.predict(Xtrain)
    stacking_test_pred = stacking_reg.predict(Xtest)

    stacking_train_mse = (mean_squared_error(ytrain, stacking_train_pred))
    stacking_test_mse = (mean_squared_error(ytest, stacking_test_pred))

    print('Train RMSE:', stacking_train_mse)
    print('Test RMSE:', stacking_test_mse)

    joblib.dump(stacking_reg, 'stacking_model.pkl')

    loaded_model = joblib.load('stacking_model.pkl')
    predictions = loaded_model.predict(Xtest)

    feature_importance(data, X_train, y_train, save_path="feature_importance_plot.png")


    return predictions

predictions = model_stacking(X_train, X_test, y_train, y_test)



def test_vs_prediction(df, y_test, predictions, save_path=None):
    plt.figure(figsize=(10, 5))

    df_dates = df['Day'][:1000]

    plt.plot(df_dates, y_test[:1000], label='Test Verileri', marker='o')
    plt.plot(df_dates, predictions[:1000], label='Tahmin Verileri', marker='x')

    plt.title('Test Verileri ve Tahmin Verileri Karşılaştırması')
    plt.xlabel('Tarih')
    plt.ylabel('Değerler')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()

test_vs_prediction(data, y_test, predictions, save_path='EnsembleML_Model_prediction_vs_Test.png')

