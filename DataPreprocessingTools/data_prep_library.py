import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sktime.performance_metrics.forecasting import mean_squared_error

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)

def quick_look(df):

    print("DataFrame Shape:", df.shape)
    print("*********************************")
    print("DataFrame Size:", df.size)
    print("*********************************")
    print("DataFrame Info:")
    print(df.info())
    print("*********************************")
    print("DataFrame Basic Statistics:")
    print(df.describe())
    print("*********************************")
    print("DataFrame Number of Unique Values:")
    print(df.nunique())
    print("*********************************")
    print("DataFrame Variable Types:")
    print(df.dtypes)
    print("Number of NaN Values:")
    print(df.isna().sum())
    print("*********************************")
    print("DataFrame Null:")
    print("Null Count:", df.isnull())

def rename_columns(df):
    df.columns = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity','Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

def variable_fix(df):
    df = df.drop(0, axis=0)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='%d/%m/%Y')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df['Hour'] = df['Time'].dt.hour
    df['Minute'] = df['Time'].dt.minute
    df = df.drop(['Date', 'Time', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], axis=1)
    numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column] = df[column].fillna(df[column].mean())
    return df

def feature_importance(df, Xtrain, ytrain, save_path=None):
    lgbm_reg = LGBMRegressor()
    lgbm_reg.fit(Xtrain, ytrain)
    feature_names = Xtrain.columns
    important_features = pd.Series(lgbm_reg.feature_importances_, index=feature_names)

    fig, ax = plt.subplots()
    important_features.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Feature Importance")
    ax.set_xlabel("Features")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def regression_test(X_train, X_test, y_train, y_test):
    """
    Bu fonksiyon farklı makine öğrenmesi modellerinin regresyon modellerinin başarısını değerlendirmek
    amacıyla yazılmıştır.

    :parameter X_train Hedef değişken olmadan data setinin 0.7 içeren değişken
    :parameter X_test  Hedef değişken olmadan data setinin 0.3 içeren değişken
    :parameter y_train Hedef değişkenin 0.7 verisini içeren değişken
    :parameter y_test  Hedef değişkenin 0.3 verisini içeren değişken

    Eğitim seti MSE: lgbm_reg 0.03450985211213242
    Test seti MSE: lgbm_reg 0.03493478718486829

    Eğitim seti R^2: lgbm_reg 0.9989206107707472
    Test seti R^2: lgbm_reg 0.9988956651931296
    Learning rate set to 0.129361

    Eğitim seti MSE catboost: 0.03362224287371401
    Test seti MSE catboost: 0.03408847011965142

    Eğitim seti R^2 catboost: 0.998975421482384
    Test seti R^2 catboost: 0.9989485234826329

    Eğitim seti MSE: linear_reg 0.04304562150595534
    Test seti MSE: linear_reg 0.043286259792402956

    Eğitim seti R^2: linear_reg 0.998975421482384
    Test seti R^2: linear_reg 0.9989485234826329

    Eğitim seti MSE: xgboost_reg 0.03352028475998553
    Test seti MSE: xgboost_reg 0.0341747183126887

    Eğitim seti R^2: xgboost_reg 0.9989816260477256
    Test seti R^2: xgboost_reg 0.9989431960111808

    LGBMRegressor, XGBRegressor test verisi üzerinde en iyi sonuç veren iki modeldir.

    """
    # Regresyon objelerinin çağırılması
    lgbm_reg = LGBMRegressor()
    catboost_reg = CatBoostRegressor()
    linear_reg = LinearRegression()
    xgboost_reg = XGBRegressor()

    # Lightgbm regresyon modelinin fitlenmesi ve tahmin alınarak kareler farkı ve ortalama kareler metriklerine göre değerlendirilmesi
    lgbm_reg.fit(X_train, y_train)
    y_train_lbgm_pred = lgbm_reg.predict(X_train)
    y_test_lbgm_pred = lgbm_reg.predict(X_test)

    train_rmse_lbgm = (mean_squared_error(y_train, y_train_lbgm_pred))
    test_rmse_lbgm = (mean_squared_error(y_test, y_test_lbgm_pred))

    train_r2_lbgm = r2_score(y_train, y_train_lbgm_pred)
    test_r2_lbgm = r2_score(y_test, y_test_lbgm_pred)

    print("Eğitim seti MSE: lgbm_reg", train_rmse_lbgm)
    print("Test seti MSE: lgbm_reg", test_rmse_lbgm)
    print("Eğitim seti R^2: lgbm_reg", train_r2_lbgm)
    print("Test seti R^2: lgbm_reg", test_r2_lbgm)

    #catboost regresyon modelinin fitlenmesi ve tahmin alınarak kareler farkı ve ortalama kareler metriklerine göre değerlendirilmesi
    catboost_reg.fit(X_train, y_train)
    y_train_catboost_pred = catboost_reg.predict(X_train)
    y_test_catboost_pred = catboost_reg.predict(X_test)

    train_rmse_catboost = (mean_squared_error(y_train, y_train_catboost_pred))
    test_rmse_catboost = (mean_squared_error(y_test, y_test_catboost_pred))

    train_r2_catboost = r2_score(y_train, y_train_catboost_pred)
    test_r2_catboost = r2_score(y_test, y_test_catboost_pred)

    print("Eğitim seti RMSE catboost:", train_rmse_catboost)
    print("Test seti RMSE catboost:", test_rmse_catboost)
    print("Eğitim seti R^2 catboost:", train_r2_catboost)
    print("Test seti R^2 catboost:", test_r2_catboost)

    # linear regresyon modelinin fitlenmesi ve tahmin alınarak kareler farkı ve ortalama kareler metriklerine göre değerlendirilmesi
    linear_reg.fit(X_train, y_train)
    y_train_lineer_reg_pred = linear_reg.predict(X_train)
    y_test_lineer_reg_pred = linear_reg.predict(X_test)

    train_rmse_lineer_reg = np.sqrt(mean_squared_error(y_train, y_train_lineer_reg_pred))
    test_rmse_lineer_reg = np.sqrt(mean_squared_error(y_test, y_test_lineer_reg_pred))

    train_r2_lineer_reg = r2_score(y_train, y_train_catboost_pred)
    test_r2_lineer_reg = r2_score(y_test, y_test_catboost_pred)

    print("Eğitim seti RMSE: linear_reg", train_rmse_lineer_reg)
    print("Test seti RMSE: linear_reg", test_rmse_lineer_reg)
    print("Eğitim seti R^2: linear_reg", train_r2_lineer_reg)
    print("Test seti R^2: linear_reg", test_r2_lineer_reg)

    #xgboost regresyon modelinin fitlenmesi ve tahmin alınarak kareler farkı ve ortalama kareler metriklerine göre değerlendirilmesi
    xgboost_reg.fit(X_train, y_train)
    y_train_xgboost_pred = xgboost_reg.predict(X_train)
    y_test_xgboost_pred = xgboost_reg.predict(X_test)

    train_rmse_xgboost_reg = np.sqrt(mean_squared_error(y_train, y_train_xgboost_pred))
    test_rmse_xgboost_reg = np.sqrt(mean_squared_error(y_test, y_test_xgboost_pred))

    train_r2_xgboost_reg = r2_score(y_train, y_train_xgboost_pred)
    test_r2_xgboost_reg = r2_score(y_test, y_test_xgboost_pred)

    print("Eğitim seti RMSE: xgboost_reg", train_rmse_xgboost_reg)
    print("Test seti RMSE: xgboost_reg", test_rmse_xgboost_reg)
    print("Eğitim seti R^2: xgboost_reg", train_r2_xgboost_reg)
    print("Test seti R^2: xgboost_reg", test_r2_xgboost_reg)





# def optimize_hyperparameters(X_train, y_train, n_trials=100, early_stopping_rounds=5):
#     best_score = float('-inf')
#     num_trials_without_improvement = 0
#
#     def objective(trial):
#         nonlocal best_score, num_trials_without_improvement
#
#         params = {
#             'n_estimators_lgbm': trial.suggest_int('n_estimators_lgbm', 50, 500),
#             'learning_rate_lgbm': trial.suggest_uniform('learning_rate_lgbm', 0.01, 0.1),
#             'max_depth_lgbm': trial.suggest_int('max_depth_lgbm', 3, 10),
#             'iterations_xgboost': trial.suggest_int('iterations_xgboost', 50, 500),
#             'eta_xgboost': trial.suggest_uniform('eta_xgboost', 0.01, 0.1),
#             'max_depth_xgboost': trial.suggest_int('max_depth_xgboost', 3, 10),
#         }
#
#         lgbm_reg = LGBMRegressor(n_estimators=params['n_estimators_lgbm'], learning_rate=params['learning_rate_lgbm'], max_depth=params['max_depth_lgbm'])
#         xgb_reg = XGBRegressor(n_estimators=params['iterations_xgboost'], eta=params['eta_xgboost'], max_depth=params['max_depth_xgboost'])
#
#         regressors = [('lgbm', lgbm_reg), ('xgb', xgb_reg)]
#         stacking_reg = StackingRegressor(estimators=regressors, final_estimator=LGBMRegressor(), n_jobs=-1)
#         pipeline = make_pipeline(StandardScaler(), stacking_reg)
#
#         kf = KFold(n_splits=5, shuffle=True, random_state=42)
#         mae = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_squared_error').mean()
#
#         # Early stopping kontrolü
#         if mae > best_score:
#             best_score = mae
#             num_trials_without_improvement = 0
#         else:
#             num_trials_without_improvement += 1
#
#         if num_trials_without_improvement >= early_stopping_rounds:
#             raise optuna.TrialPruned()
#
#         return mae
#
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=n_trials)
#
#     return study
#
# optimize_hyperparameters(X_train, y_train, n_trials=30, early_stopping_rounds=5)




