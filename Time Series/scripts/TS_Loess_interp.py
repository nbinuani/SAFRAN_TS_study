import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import *
import statsmodels.tsa.api as smt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.nonparametric.smoothers_lowess import lowess
import plotly.offline as py
import plotly.graph_objs as go
from scipy.interpolate import interp1d
from scipy.signal.wavelets import cwt


def ts_plot(y, lags=None, title=''):
   '''
   Calculate acf, pacf, histogram, and qq plot for a given time series
   '''
   # if time series is not a Series object, make it so
   if not isinstance(y, pd.Series):
      y = pd.Series(y)

   # initialize figure and axes
   fig = plt.figure(figsize=(14, 12))
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
   smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
   smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)

   # qq plot
   qqplot(y, line='s', ax=qq_ax)
   qq_ax.set_title('Normal QQ Plot')

   # hist plot
   y.plot(ax=hist_ax, kind='hist', bins=25)
   hist_ax.set_title('Histogram')
   plt.tight_layout()
   plt.show()
   return

def _retirerTendance(data):
    x = [i for i in range(0, len(data))]
    data_no_trend = [i for i in range(0, len(data))]
    sum_x = 0
    sum_y = 0
    sum_xx = 0
    sum_xy = 0
    for k in range(0, len(data)):
        sum_x += x[k]
        sum_y += data[k]
        sum_xx += pow(x[k], 2)
        sum_xy += (x[k] * data[k])
    slope = (len(data)*sum_xy - sum_x * sum_y) / (len(data) * sum_xx - sum_x * sum_x)
    for j in range(0, len(data)):
        data_no_trend[j] = data[j] - slope*j
    return data_no_trend, slope

def _LOESS(data, frac):
    lowess_ts = lowess(endog=data.values, exog=data.index, return_sorted=False, frac=frac)
    lowess_TS = pd.Series(index=data.index, data=lowess_ts)
    # plt.scatter(x=data.index, y=data.values)
    # plt.plot(lowess_TS, label='Lissage LOESS', color='red')
    # plt.legend()
    # plt.show()
    return lowess_TS

def _error(original, create):
    error = mean_squared_error(original, create)
    print('MSE prediction : ' + str(error))
    return error

def main():
    # DATA load
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    path = r'D:\users\S598658\Projects\Flat_pattern\env\Data_production_variations.csv'
    data = pd.read_csv(path, sep=',', parse_dates=['TimeIndex'], index_col='TimeIndex', date_parser=dateparse)
    data = data[102:162]
    ts = data["Data"]
    last_element = ts.index[-1]
    nb_predictions = 10
    SPLIT = 1 - (nb_predictions/len(ts))
    TRAIN_SIZE = int(len(ts) * SPLIT)
    ts_train, ts_test = ts[0:TRAIN_SIZE ], ts[TRAIN_SIZE:len(ts)]

    train = np.array(ts_train)
    test = np.array(ts_test)
    index_test = ts_test.index
    ts_test.reset_index(drop=True, inplace=True)

###############

    # Modeling & Prediction DATA
    signal_no_tendance, coef_tendance = _retirerTendance(train)
    FRAC =  1/20
    coefficients = _LOESS(ts_train, frac=FRAC)
    WINDOW = len(ts_test)
    coefficients = coefficients[len(coefficients)- WINDOW:]
    index_train = coefficients.index
    TS_train = pd.Series(index=index_train, data=coefficients.values)

    first_coef = TS_train.values[1]
    last_coef = TS_train.values[-1]
    delta = last_coef - first_coef
    DELTA = 0.1
    coefficients.reset_index(drop=True, inplace=True)
    print('delta :')
    print(delta)
    f = interp1d(coefficients.index, coefficients.values)
    if (abs(delta) > DELTA):
        y_predictions = f(ts_test.index) + delta
        print('ajout delta')
    else:
        y_predictions = f(ts_test.index)

    TS_pred = pd .Series(index=index_test, data=y_predictions)
    TS_final = pd.concat([TS_train, TS_pred], axis=0)

    mean_signal = int(sum(TS_pred)) / len(TS_pred)
    std = np.std(TS_pred)
    born_sup = TS_pred + (mean_signal - 1.96 * std) / np.sqrt(len(test))
    born_inf = TS_pred - (mean_signal - 1.96 * std) / np.sqrt(len(test))

    # Estimation error on predictions
    _error(ts_test, TS_pred)

############

    # Plotting results
    trace0 = go.Scatter(
        x=data.index,
        y=data.Data,
        mode='markers',
        name='TS'
    )
    trace1 = go.Scatter(
        x=TS_train.index,
        y=TS_train,
        mode='lines',
        name='Model Fourier'
    )
    trace2 = go.Scatter(
        x=TS_pred.index,
        y=TS_pred,
        mode='lines',
        name='Predictions Loess & interpolation'
    )
    trace3 = go.Scatter(
        x=TS_pred.index,
        y=born_sup,
        name='Born Sup',
        line=dict(
            color='gray',
            dash='dot')
    )
    trace4 = go.Scatter(
        x=TS_pred.index,
        y=born_inf,
        name='Born Inf',
        line=dict(
            color='gray',
            dash='dot'
        )
    )
    trace5 = go.Scatter(
        x=TS_pred.index,
        y=born_sup,
        fill='tonexty',
        mode='none',
        name='IC 95%'
    )
    layout = go.Layout({
        'shapes': [
            {
                'type': 'line',
                'x0': '2015-01-01',
                'y0': 1,
                'x1': last_element,
                'y1': 1,
                'line': {
                    'color': 'red',
                }
            },
            {
                'type': 'line',
                'x0': '2015-01-01',
                'y0': -1,
                'x1': last_element,
                'y1': -1,
                'line': {
                    'color': 'red'
                }
            }
        ]
    })

    donnees = [trace0, trace1, trace2, trace3, trace4, trace5]
    fig = dict(data=donnees, layout=layout)
    py.plot(fig, filename='TS_Loess_interp.html')

if __name__ == '__main__':
    main()