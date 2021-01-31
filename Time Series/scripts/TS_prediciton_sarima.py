# -*- coding utf-8 -*-
"""
MAJ : 05/10/18
S598658 Binuani Nicolas
"""
# % Librairies
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.graphics.gofplots import qqplot
import plotly.offline as py
import plotly.graph_objs as go
from scipy.stats import linregress
from arch import arch_model
from tkinter.filedialog import askopenfilename

def test_stationarity(timeseries, wd):
    # Perform Dickey-Fuller Test
    print("Result of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    p_value = dfoutput['Test Statistic']
    alpha = dfoutput['Critical Value (1%)']
    stationnary = False
    if p_value < alpha:
        stationnary = True
    return dfoutput, stationnary

def stationnary_coef(data, WD, type):
    print('Test Stationanry '+str(type))
    data_trend_sta, data_stationary = test_stationarity(data, wd=WD)
    differenciation = 0
    if data_stationary == False:
        differenciation = 1
    return differenciation

def _differenciation(ts, coef_diff):
    ts_diff = ts - ts.shift(coef_diff)
    return ts_diff

############ SARIMA ################################

def evaluate_models(dataset, d, d_seasonal, TRAIN_SIZE, WD):
    train, test = dataset[0:TRAIN_SIZE + 1], dataset[TRAIN_SIZE:len(dataset)]
    best_score = float(10000)
    best_cfg_arima = (1, d, 0)
    best_cfg_seasonal = (1, d_seasonal, 0, WD)
    p_values = range(1, 3)
    q_values = range(1, 2)
    for p in p_values:
        for q in q_values:
            for p_s in p_values:
                for q_s in q_values:
                    order = (p, d, q)
                    seasonal_order = (p_s, d_seasonal, q_s, WD)
                    try:
                        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
                        md_fit = model.fit()
                        aic = SARIMAXResults.aic(md_fit)
                        if abs(aic) < best_score:
                            best_score, best_cfg_arima, best_cfg_seasonal = aic, order, seasonal_order
                    except:
                        continue
    print('Best ARIMA%s Best Seasonal%s AIC=%.3f' % (best_cfg_arima, best_cfg_seasonal, best_score))
    return best_cfg_arima, best_cfg_seasonal, best_score

def test_model_sarima(dataset, d, d_seasonal, TRAIN_SIZE, WD):
    warnings.filterwarnings("ignore")
    data_train, data_test = dataset[0:TRAIN_SIZE + 1], dataset[TRAIN_SIZE:len(dataset)]
    order_arima, order_seasonal, cfg = evaluate_models(dataset, d=d,
                                                        d_seasonal=d_seasonal,
                                                        TRAIN_SIZE=TRAIN_SIZE, WD=WD)
    model = SARIMAX(data_train, order=order_arima,seasonal_order=order_seasonal)
    md_fit = model.fit()
    ts_arima = pd.Series(md_fit.fittedvalues)
    predictions = md_fit.predict(start=TRAIN_SIZE, end=TRAIN_SIZE + len(data_test) -1)
    print('(S)ARIMA model summary :')
    print(md_fit.summary())
    return ts_arima, predictions


############################################
############### GARCH ######################

def evaluate_garch_model(dataset, TRAIN_SIZE, split_time):
    warnings.filterwarnings("ignore")
    train, test = dataset[0:TRAIN_SIZE + 1], dataset[TRAIN_SIZE:len(dataset)]
    best_score = float(10000)
    best_p = (1)
    best_q = int(1)
    p_values = range(1, 3)
    q_values = range(1, 2)
    for p in p_values:
        for q in q_values:
            try:
                am = arch_model(dataset, vol='Garch', p=p, q=q)
                res = am.fit(last_obs=split_time)
                aic = SARIMAXResults.aic(res)
                if abs(aic) < best_score:
                    best_score, best_p, best_q = aic, p, q
                    # print('GARCH : p=%i  GARCH : q=%i  AIC=%.3f' % (p, q, aic))
            except:
                continue
    # print('Best config GARCH model p=%i q=%i AIC=%.3f' % (best_p, best_q, best_score))

    am = arch_model(dataset, vol='Garch', p=best_p, q=best_q)
    res = am.fit(last_obs=split_time)
    # print('GARCH model summary :')
    # print(res.summary())
    forecast_analytic = res.forecast(horizon=len(test), method='analytic')
    forecast_variance = forecast_analytic.variance[split_time:]
    forecast_volatility = forecast_analytic.residual_variance[split_time:]
    forecast_garch = forecast_variance.iloc[:, 0]
    forecast_vol = forecast_volatility.iloc[:, 0]
    return forecast_garch, forecast_vol

#######################################
########### FINAL #####################

def predictions_final(pred_sarima, forecast_garch):
    final_predictions = (pred_sarima + forecast_garch)/2
    return final_predictions

##### Drift Fonction ########

def _mydrift(predictions, lim_sup, lim_inf):
    #color = ['red', 'orange', 'green']
    color = ['rgb(215, 11, 11)', 'rgb(240, 140, 0)', 'rgb(0, 204, 0)']
    nb_pts_critic = int(len(predictions)/2)
    clign = color[2]
    if predictions[: nb_pts_critic].mean() > lim_sup or predictions[: nb_pts_critic].mean() < lim_inf :
        clign = color[0]
    elif predictions[nb_pts_critic:].mean() > lim_sup or predictions[nb_pts_critic:].mean() < lim_inf :
        clign = color[1]
    # print('Code couleur des avertissements :')
    # print('Rouge = premières valeurs de prédiction au dessus des limites = état critique')
    # print('Orange = dernières valeurs de prédiction au dessus des limites = avertissement')
    # print('Vert = RAS')
    # print(clign)
    return clign

def _error(original, create):
    error = mean_squared_error(original, create)
    print('MSE prediction : ' + str(error))
    return error

def _loading_data(path, nb_predictions, start, stop):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    data = pd.read_csv(path, sep=',', parse_dates=['TimeIndex'], index_col='TimeIndex', date_parser=dateparse)
    data = data[start:stop]
    ts = data["Data"]
    SPLIT = 1 - (nb_predictions / len(ts))
    TRAIN_SIZE = int(len(ts) * SPLIT)
    WD = int(len(data) / 30)  # window for test stationnary
    ts_train, ts_test = ts[0:TRAIN_SIZE + 1], ts[TRAIN_SIZE:len(ts)]

    return ts_train, ts_test, ts, data, WD, TRAIN_SIZE

def _my_reglin(ts_test, ts_SARIMA):
    LIN_TRAIN = len(ts_test)
    x = [l for l in range(len(ts_SARIMA))]
    x_pred = x[len(x) - LIN_TRAIN:]
    ts_SARIMA_lin = ts_SARIMA[len(ts_SARIMA) - LIN_TRAIN:]
    ## Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x=x_pred, y=ts_SARIMA_lin.values)
    trend = []
    for w in range(len(x_pred)):
        trend.append(slope * x_pred[w] + intercept)
    return trend

def _plot(data, sarima, predictions, variability, color, trend_linear, born_inf, born_sup, limit_sup, limit_inf):
    first_element = data.index[0]
    last_element = data.index[-1]
    #Plotting results
    trace0 = go.Scatter(
        x=data.index,
        y=data.Data,
        mode='markers',
        name='TS'
    )
    trace1 = go.Scatter(
        x=sarima.index,
        y=sarima,
        mode='lines',
        name='Train Set',
        marker=dict(
            color='rgb(11, 133, 215)'
        )
    )
    trace2 = go.Scatter(
        x=predictions.index,
        y=predictions,
        mode='markers+lines',
        name='Test Set'
    )
    trace3 = go.Scatter(
        x=predictions.index,
        y=born_sup,
        name='Born Sup',
        line=dict(
            color='gray',
            dash='dot')
    )
    trace4 = go.Scatter(
        x=predictions.index,
        y=born_inf,
        name='Born Inf',
        line=dict(
            color='gray',
            dash='dot'
        )
    )
    trace5 = go.Scatter(
        x=predictions.index,
        y=born_sup,
        fill='tonexty',
        mode='none',
        name='IC 95%'
    )
    trace6 = go.Scatter(
        x=predictions.index,
        y=trend_linear,
        mode='lines',
        name='Reg lin TREND',
        marker=dict(
            size=5,
            color=color),
        line=dict(
            width=5,
            color=color
            )
    )
    trace7 = go.Scatter(
        x=predictions.index,
        y=variability,
        mode='markers+lines',
        name='Variability predictions',
        marker=dict(
            color='rgb(100, 100, 100)'
        )
    )
    layout = go.Layout({
        'shapes': [
            {
                'type': 'line',
                'x0': first_element,
                'y0': limit_sup,
                'x1': last_element,
                'y1': limit_sup,
                'line': {
                    'color': 'red',
                }
            },
            {
                'type': 'line',
                'x0': first_element,
                'y0': limit_inf,
                'x1': last_element,
                'y1': limit_inf,
                'line': {
                    'color': 'red'
                }
            }
        ]
    })
    donnees = [trace0, trace1, trace2, trace3, trace4, trace5, trace6]
    fig = dict(data=donnees, layout=layout)
    py.plot(fig, filename='TS_sarima.html')

def _parameters_entries():
    filepath = askopenfilename(title="Choisir le fichier csv", filetypes=[('csv files', '.csv'), ('all files', '.*')])
    nb_predictions = int(input("Number of predictions day :"))
    limit_sup = float(input("Limit sup of the signal :"))
    limit_inf = float(input("Limit inf of the signal :"))
    start = int(input("Signal start (int) :"))
    stop = int(input("Signal stop (int) :"))
    return filepath, nb_predictions, limit_sup, limit_inf, start, stop

def main():
    # Load DATA
    path, nb_predictions, limit_sup, limit_inf, start, stop = _parameters_entries()
    ts_train, ts_test, ts, data, WD, TRAIN_SIZE = _loading_data(path, nb_predictions, start, stop)
    print(ts_train)
    print(ts_test)
    d = stationnary_coef(ts_train, WD, type='Time Series')

    ts_sarima, predictions_sarima = test_model_sarima(ts, d=d, d_seasonal=d, WD=WD, TRAIN_SIZE=TRAIN_SIZE)
    model_sarima = pd.Series(index=ts_train.index, data=ts_sarima)
    pred_sarima = pd.Series(index=ts_test.index, data=predictions_sarima)

    ts_SARIMA = pd.concat([model_sarima, pred_sarima], axis=0)
    ts_SARIMA.drop_duplicates(inplace=True)

    trend = _my_reglin(ts_test, ts_SARIMA)

    avertissement_color = _mydrift(pred_sarima, limit_sup, limit_inf)
    print('Couleur regression !!!!!!!!!!')
    print(avertissement_color)
    split_date = ts_train.index[-1]
    forecast, volatility = evaluate_garch_model(ts, TRAIN_SIZE=TRAIN_SIZE, split_time=split_date)
    print('Volatility model')
    print(forecast)
    print(volatility)
    pred_final = predictions_final(pred_sarima, forecast)

    mean_pred = int(sum(pred_sarima)) / len(pred_sarima)
    std_pred = np.std(pred_sarima)
    born_sup = pred_sarima + (mean_pred + 1.96 * std_pred) / np.sqrt(len(ts_test))
    born_inf = pred_sarima - (mean_pred + 1.96 * std_pred) / np.sqrt(len(ts_test))

    # Estimation error on predictions
    _error(ts_test, pred_sarima)

    # Plot results
    print('TS TRAIN :')
    print(ts_train)
    print(len(ts_train))
    print('TS TEST :')
    print(ts_test)
    print(len(ts_test))
    print('DATA :')
    print(data)
    print(len(data))

    _plot(data, sarima=model_sarima, predictions=pred_sarima, trend_linear=trend, color=avertissement_color,
          born_inf=born_inf, born_sup=born_sup, limit_sup=limit_sup, limit_inf=limit_inf, variability=forecast)

if __name__ == '__main__':
    main()

