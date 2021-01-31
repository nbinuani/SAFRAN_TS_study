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
from statsmodels.tsa.stattools import adfuller
from scipy.stats import linregress

def ts_plot(y, lags=None, title=''):
   '''
   Calculate acf, pacf, histogram, and qq plot for a given time series
   '''
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
    return lowess_TS

# Directement la fonction np.fft.fftfreq retourne un array des fréquence d'échantillonage
def _fftfreq(n):
    val = 1.0/n
    N = floor((n-1)/2)+1
    results = [i for i in range(0,int(N))]
    p1 = [i for i in range(0, int(n))]
    for k in range(0, int(N)):
        results[k] = k*val
    for j in range(0, int(n)):
        results[j] = -floor(n/2) - (N-j)*val
    return results

def _tri_indice_freq(freqs):
    indice = [i for i in range(0, int(len(freqs)))]
    tmpTab = pd.DataFrame({'freqs': freqs, 'indice': indice})
    tmpTab['abs'] = abs(tmpTab.freqs)
    tmpTab.sort_values(by='abs', inplace=True)
    return tmpTab.indice

def _prediction(horizon, nb_harm, domaine_freq, f, indices_tries, coef_tendance):
    signal_restaure = [0 for i in range(0,(len(domaine_freq)+horizon -1))]
    for k in range(0, 1+nb_harm*2):
        indice = indices_tries[k]
        amplitude = sqrt( pow(domaine_freq[indice].real, 2) + pow(domaine_freq[indice].imag, 2)) / len(domaine_freq)
        phase = atan2(domaine_freq[indice].imag, domaine_freq[indice].real)
        facteur = 2*pi*f[indice]
        for j in range(0, (len(domaine_freq)+horizon-1)):
            signal_restaure[j] += amplitude*(sin(facteur*j+phase))
    return signal_restaure

def _comb_signal(signal_restaure, coef_tendance):
    for l in range(len(signal_restaure)):
        signal_restaure[l] = signal_restaure[l] + coef_tendance[l]
    return signal_restaure

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
    # DATA load
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    path = r'D:\users\S598658\Projects\Flat_pattern\env\Data_production_variations.csv'
    data = pd.read_csv(path, sep=',', parse_dates=['TimeIndex'], index_col='TimeIndex', date_parser=dateparse)
    data = data[42:102]
    ts = data["Data"]
    last_element = ts.index[-1]
    nb_predictions = 10
    SPLIT = 1 - (nb_predictions/len(ts))
    TRAIN_SIZE = int(len(ts) * SPLIT)
    ts_train, ts_test = ts[0:TRAIN_SIZE], ts[TRAIN_SIZE:len(ts)]
    ts_train_fft, ts_test_fft = ts[0:TRAIN_SIZE +1], ts[TRAIN_SIZE:len(ts)]
    ts_diff = pd.Series(index=ts_train.index)


    d_TS = stationnary_coef(ts_train, type='TS')
    if d_TS >= 1:
        ts_diff = ts_train - ts_train.shift(d_TS)
    index_test = ts_test.index
    ts_test.reset_index(drop=True, inplace=True)
    print(d_TS)
###############
    # Modeling & Prediction DATA
    # LOESS
    FRAC = 1/5
    coefficients_train = _LOESS(ts_train,frac=FRAC)
    index_train = coefficients_train.index
    WINDOW = len(ts_test)
    coefficients_wd = coefficients_train[len(coefficients_train)- WINDOW:]
    coefficients_wd.reset_index(drop=True, inplace=True)
    TS_train = pd.Series(index=index_train, data=coefficients_train.values)
    f = interp1d(coefficients_wd.index, coefficients_wd.values)
    first_coef = coefficients_wd.values[1]
    last_coef = coefficients_wd.values[-1]
    delta = last_coef - first_coef
    DELTA = 0.1
    print('delta :')
    print(delta)
    print('TS TEST')
    print(ts_test)
    if (abs(delta) > DELTA):
        y_predictions = f(ts_test.index) + delta
        print("ajout delta")
    else:
        y_predictions = f(ts_test.index)

    TS_test = pd .Series(index=index_test, data=y_predictions)
    TS_loess = pd.concat([TS_train, TS_test], axis=0)

    # FFT
    train = np.array(ts_train_fft)
    test = np.array(ts_test_fft)
    signal_no_tendance, coef_tendance = _retirerTendance(train)
    domaine_freq = np.fft.fft(signal_no_tendance)
    f = np.fft.fftfreq(len(ts_train_fft))
    indices_tries = _tri_indice_freq(f)
    horizon = len(ts_test_fft)
    harm_div = 10
    nb_harm = int(len(data) / harm_div)
    print('Coef pred tendance')
    print(coef_tendance)
    signal_pred = _prediction(horizon=horizon, nb_harm=nb_harm, domaine_freq=domaine_freq,
                              f=f, indices_tries=indices_tries, coef_tendance=coefficients_train)
    array_final = _comb_signal(signal_pred, TS_loess.values)

    TS_pred = pd.Series(index=data.index, data=array_final)
    TS_pred_train = TS_pred[0:TRAIN_SIZE]
    TS_pred_test = TS_pred[TRAIN_SIZE:len(data)]

    mean_signal = int(sum(TS_pred_test)) / len(TS_pred_test)
    std = np.std(TS_pred_test)
    born_sup = TS_test + (mean_signal - 1.96 * std) / np.sqrt(len(test))
    born_inf = TS_test - (mean_signal - 1.96 * std) / np.sqrt(len(test))

    # Estimation error on predictions
    _error(ts_test, TS_pred_test)

    limite_sup = 1
    limite_inf = -1

    LIN_TRAIN = len(ts_test)
    x = [l for l in range(len(TS_pred))]
    x_pred = x[len(x) - LIN_TRAIN:]
    TS_pred_lin = TS_pred[len(TS_pred) - LIN_TRAIN:]
    ## Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x=x_pred, y=TS_pred_lin.values)
    trend = []
    for w in range(len(x_pred)):
        trend.append(slope * x_pred[w] + intercept)
    avertissement_color = _mydrift(TS_pred_lin, limite_sup, limite_inf)

############

    # Plotting results
    trace0 = go.Scatter(
        x=data.index,
        y=data.Data,
        mode='markers',
        name='TS'
    )
    trace1 = go.Scatter(
        x=TS_pred_train.index,
        y=TS_pred_train,
        mode='lines',
        name='Train set'
    )
    trace2 = go.Scatter(
        x=TS_pred_test.index,
        y=TS_pred_test,
        mode='lines',
        name='Test set'
    )
    trace3 = go.Scatter(
        x=TS_test.index,
        y=born_sup,
        name='Born Sup',
        line=dict(
            color='gray',
            dash='dash')
    )
    trace4 = go.Scatter(
        x=TS_test.index,
        y=born_inf,
        name='Born Inf',
        line=dict(
            color='gray',
            dash='dash'
        )
    )
    trace5 = go.Scatter(
        x=TS_test.index,
        y=born_sup,
        fill='tonexty',
        mode='none',
        name='IC 95%',
    )
    trace6 = go.Scatter(
        x=TS_pred_test.index,
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
    donnees = [trace0, trace1, trace2, trace3, trace4, trace5, trace6]
    fig = dict(data=donnees, layout=layout)
    py.plot(fig, filename='TS_FFT_Loess_interp.html')

if __name__ == '__main__':
    main()