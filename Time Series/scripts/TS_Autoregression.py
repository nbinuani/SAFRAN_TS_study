import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.ar_model import ARResults
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress

def ts_plot(y, lags=None, title=''):
   '''
   Calculate acf, pacf, histogram, and qq plot for a given time series
   '''
   # if time series is not a Series object, make it so
   if not isinstance(y, pd.Series):
      y = pd.Series(y)

   # initialize figure and axes
   fig = plt.figure(figsize=(10, 8))
   layout = (3, 2)
   ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
   acf_ax = plt.subplot2grid(layout, (1, 0))
   pacf_ax = plt.subplot2grid(layout, (1, 1))
   qq_ax = plt.subplot2grid(layout, (2, 0))
   hist_ax = plt.subplot2grid(layout, (2, 1))

   # time series plot
   y.plot(ax=ts_ax)
   plt.legend(loc='best')
   ts_ax.set_title(title)

   # acf and pacf
   plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
   plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
   # smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
   # smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)

   # qq plot
   qqplot(y, line='s', ax=qq_ax)
   qq_ax.set_title('Normal QQ Plot')

   # hist plot
   y.plot(ax=hist_ax, kind='hist', bins=25)
   hist_ax.set_title('Histogram')
   plt.tight_layout()
   plt.show()
   return

def _my_AR(ts, SPLIT):
    # create lagged dataset
    dataframe = pd.concat([ts.shift(1), ts], axis=1)
    dataframe.columns = ['t-1', 't+1']
    # split into train and test sets
    X = dataframe.values
    train_size = int(len(X) *SPLIT)
    train, test = X[1:train_size], X[train_size:]
    train_X, train_y = train[:, 0], train[:, 1]
    test_X, test_y = test[:, 0], test[:, 1]
    # persistence model on training set
    train_pred = [x for x in train_X]
    # calculate residuals
    train_resid = [train_y[i] - train_pred[i] for i in range(len(train_pred))]
    # model the training set residuals
    model = AR(train_resid)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params

    # walk forward over time steps in test
    history = train_resid[len(train_resid) - window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    expected_error = list()
    for t in range(len(test_y)):
        # persistence
        yhat = test_X[t]
        error = test_y[t] - yhat
        expected_error.append(error)
        # predict error
        length = len(history)
        lag = [history[i] for i in range(length - window, length)]
        pred_error = coef[0]
        for d in range(window):
            pred_error += coef[d + 1] * lag[window - d - 1]
        predictions.append(-1*pred_error)
        history.append(error)
        print('predicted error=%f, expected error=%f' % (pred_error, error))
    # plot predicted error
    # plt.plot(expected_error)
    # plt.plot(predictions, color='red')
    # plt.show()
    AR_train = model_fit.fittedvalues
    return predictions, AR_train

def test_stationarity(timeseries):
    # Perform Dickey-Fuller Test
    print("Result of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    p_value = dfoutput['Test Statistic']
    alpha = dfoutput['Critical Value (1%)']
    stationnary = bool
    if p_value < alpha :
        stationnary = True
    else:
        stationnary = False
    return dfoutput, stationnary

def stationnary_coef(data, type):
    print('Test Stationanry '+str(type))
    data_trend_sta, data_stationary = test_stationarity(data)
    differenciation = 0
    if data_stationary == False:
        differenciation = 1
    return differenciation

def _differenciation(ts, coef_diff):
    ts_diff = ts - ts.shift(coef_diff)
    return ts_diff

def _mydrift(predictions, lim_sup, lim_inf):
    #color = ['red', 'orange', 'green']
    color = ['rgb(215, 11, 11)', 'rgb(240, 140, 0)', 'rgb(0, 204, 0)']
    nb_pts_critic = int(len(predictions)/2)
    clign = color[2]
    if predictions[: nb_pts_critic].mean() > lim_sup or predictions[: nb_pts_critic].mean() < lim_inf :
        clign = color[0]
    elif predictions[nb_pts_critic:].mean() > lim_sup or predictions[nb_pts_critic:].mean() < lim_inf :
        clign = color[1]

    print('Code couleur des avertissements :')
    print('Rouge = premières valeurs de prédiction au dessus des limites = état critique')
    print('Orange = dernières valeurs de prédiction au dessus des limites = avertissement')
    print('Vert = RAS')
    print(clign)
    return clign

def _error(original, create):
    error = mean_squared_error(original, create)
    print('MSE prediction : ' + str(error))
    return error

def main():
    # Load DATA
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    path = r'D:\users\S598658\Projects\Flat_pattern\env\Data_production_variations.csv'
    data = pd.read_csv(path, sep=',', parse_dates=['TimeIndex'], index_col='TimeIndex', date_parser=dateparse)
    data = data[70:130]
    ts = data["Data"]
    last_element = ts.index[-1]
    nb_predictions = 20
    SPLIT = 1 - (nb_predictions/len(ts))
    WD = int(len(data)/30)  # window for test stationnary
    print('WD : %i' % WD)
    TRAIN_SIZE = int(len(ts) * SPLIT)
    ts_train, ts_test = ts[0:TRAIN_SIZE + 1], ts[TRAIN_SIZE:len(ts)]

    # Decompostition
    decomposition = seasonal_decompose(ts, freq=WD)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    residual.fillna(residual.mean(), inplace=True)
    trend.fillna(trend.mean(), inplace=True)

    d = stationnary_coef(ts, 'TS')
    ts = _differenciation(ts, d)
    ts.fillna(ts.mean(), inplace=True)
    predictions, ar = _my_AR(ts, SPLIT)
    # predictions = -1*predictions
    my_AR_pred = pd.Series(index=ts_test.index, data=predictions)

    model = AR(ts_train, freq='D', dates=ts_train.index)
    #model = AR(ts_train)
    model_fit = model.fit(ic='aic')
    predictions = model_fit.predict(start=TRAIN_SIZE, end=TRAIN_SIZE + len(ts_test)-1)
    AR_model = pd.Series(index=ts_train.index, data=model_fit.fittedvalues)
    AR_pred = pd.Series(index=ts_test.index, data=predictions)
    AR_final = pd.concat([AR_model, AR_pred], axis=0)

    mean_my_AR = int(sum(my_AR_pred)) / len(my_AR_pred)
    std_my_AR = np.std(my_AR_pred)
    my_AR_born_sup = my_AR_pred + (mean_my_AR - 1.96 * std_my_AR) / np.sqrt(len(ts_test))
    my_AR_born_inf = my_AR_pred - (mean_my_AR - 1.96 * std_my_AR) / np.sqrt(len(ts_test))

    mean_AR = int(sum(AR_pred)) / len(AR_pred)
    std_AR = np.std(AR_pred)
    AR_born_sup = AR_pred + (mean_AR - 1.96 * std_AR) / np.sqrt(len(ts_test))
    AR_born_inf = AR_pred - (mean_AR - 1.96 * std_AR) / np.sqrt(len(ts_test))

    # Estimation error on predictions
    print("my AR MSE : ")
    _error(ts_test, my_AR_pred)
    print("AR MSE : ")
    _error(ts_test, AR_pred)

    limite_sup = 1
    limite_inf = -1

    LIN_TRAIN = len(ts_test)
    x = [l for l in range(len(AR_final))]
    x_pred = x[len(x) - LIN_TRAIN:]
    AR_final_lin = AR_final[len(AR_final) - LIN_TRAIN:]
    ## Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x=x_pred, y=AR_final_lin.values)
    trend = []
    for w in range(len(x_pred)):
        trend.append(slope * x_pred[w] + intercept)
    avertissement_color = _mydrift(AR_pred, limite_sup, limite_inf)

    ############

    # Plotting results
    trace0 = go.Scatter(
        x=data.index,
        y=data.Data,
        mode='markers',
        name='TS'
    )
    trace2 = go.Scatter(
        x=my_AR_pred.index,
        y=my_AR_pred,
        mode='lines',
        name='my_AR Test Set'
    )
    trace3 = go.Scatter(
        x=my_AR_pred.index,
        y=my_AR_born_sup,
        name='Born Sup',
        line=dict(
            color='gray',
            dash='dot')
    )
    trace4 = go.Scatter(
        x=my_AR_pred.index,
        y=my_AR_born_inf,
        name='Born Inf',
        line=dict(
            color='gray',
            dash='dot'
        )
    )
    trace5 = go.Scatter(
        x=my_AR_pred.index,
        y=my_AR_born_sup,
        fill='tonexty',
        mode='none',
        name='IC 95%'
    )
    trace6 = go.Scatter(
        x=AR_model.index,
        y=AR_model,
        mode='lines',
        name='AR Train Set'
    )
    trace7 = go.Scatter(
        x=AR_pred.index,
        y=AR_pred,
        mode='lines',
        name='AR Test Set'
    )
    trace8 = go.Scatter(
        x=AR_pred.index,
        y=AR_born_sup,
        name='Born Sup',
        line=dict(
            color='gray',
            dash='dot')
    )
    trace9 = go.Scatter(
        x=AR_pred.index,
        y=AR_born_inf,
        name='Born Inf',
        line=dict(
            color='gray',
            dash='dot'
        )
    )
    trace10 = go.Scatter(
        x=AR_pred.index,
        y=AR_born_sup,
        fill='tonexty',
        mode='none',
        name='IC 95%'
    )
    trace11 = go.Scatter(
        x=AR_pred.index,
        y=trend,
        mode='lines',
        name='Reg lin TREND',
        marker=dict(
            size=5,
            color=avertissement_color),
        line=dict(
            width=5,
            color=avertissement_color
        )
    )
    layout = go.Layout({
        'shapes': [
            {
                'type': 'line',
                'x0': '2015-01-01',
                'y0': limite_sup,
                'x1': last_element,
                'y1': limite_sup,
                'line': {
                    'color': 'red',
                }
            },
            {
                'type': 'line',
                'x0': '2015-01-01',
                'y0': limite_inf,
                'x1': last_element,
                'y1': limite_inf,
                'line': {
                    'color': 'red'
                }
            }
        ]
    })
    donnees = [trace0, trace2, trace3, trace4, trace5]
    fig = dict(data=donnees, layout=layout)
    py.plot(fig, filename='TS_my_AR.html')

    donnees_1 = [trace0, trace6, trace7, trace8, trace9, trace10]
    fig = dict(data=donnees_1, layout=layout)
    py.plot(fig, filename='TS_AR.html')

if __name__ == '__main__':
    main()
