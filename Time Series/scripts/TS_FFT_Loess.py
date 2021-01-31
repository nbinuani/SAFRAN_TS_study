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
from scipy.stats import anderson
from scipy.stats import shapiro

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
    plt.scatter(x=data.index, y=data.values)
    plt.plot(lowess_TS, label='Lissage LOESS', color='red')
    plt.legend()
    plt.show()
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

# # Coef_tendance from train & test so no prediction
# def _comb_signal(signal_restaure, coef_tendance):
#     for l in range(len(signal_restaure)):
#         signal_restaure[l] = signal_restaure[l] + coef_tendance[l]
#     return signal_restaure

def _comb_signal(signal_restaure, coef_tendance):
    for l in range(len(signal_restaure)):
        signal_restaure[l] = signal_restaure[l] + coef_tendance
    return signal_restaure

def _error(original, create):
    error = mean_squared_error(original, create)
    diff_percent = sum(abs(original - create))
    # MAPE = (diff_percent / len(original)) * 100
    print('MSE prediction : ' + str(error))
    #print("MAPE : %.2f" % MAPE)
    return error

def _optimisation_parameters(data, train_size, train, nb_total_test, horizon, domaine_freq, f, indices_tries, coef_tendance):
    data_signal = list(data.Data)
    best_score, nb_harm = float("inf"), None
    best_score, nb_harm = 0, 0
    for nb_test in range(100, nb_total_test):
        signal_pred = _prediction(horizon=horizon, nb_harm=nb_test, domaine_freq=domaine_freq,
                                  f=f, indices_tries=indices_tries, coef_tendance=coef_tendance)
        signal_pred_train = signal_pred[0:train_size+1]
        mse = mean_squared_error(train, signal_pred_train)
        if mse > best_score:
            best_score = mse
            nb_harm = nb_test
            print('MSE%s Nombre harm=%.3f' % (best_score, nb_harm))
        else:
            print('pas supérieur')
    print('MSE%s Nombre harm=%.3f' % (best_score, nb_harm))
    return

def main():
    # DATA load
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    path = r'D:\users\S598658\Projects\Flat_pattern\env\Data_production_variations.csv'
    data = pd.read_csv(path, sep=',', parse_dates=['TimeIndex'], index_col='TimeIndex', date_parser=dateparse)
    data = data[0:110]
    ts = data["Data"]
    last_element = ts.index[-1]
    SPLIT = 0.91
    TRAIN_SIZE = int(len(ts) * SPLIT)
    ts_train, ts_test = ts[0:TRAIN_SIZE + 1], ts[TRAIN_SIZE:len(ts)]

    train = np.array(ts_train)
    test = np.array(ts_test)
    # ts_plot(ts, title='TS')
    # ts_plot(ts_train, title='TS train')

    # A_test, critical, significant = anderson(ts, dist='logistic')
    # print('Anderson value test')
    # print(A_test)
    # print('Critical value tested')
    # print(critical)
    # print('Sifnicant value test')
    # print(significant)
    #
    # W_test, p_value = shapiro(ts)
    # print('Shapiro value test')
    # print(W_test)
    # print('Hypothese p_value')
    # print(p_value)

###############

    # Modeling & Prediction DATA
    signal_no_tendance, coef_tendance = _retirerTendance(train)
    FRAC =  1/25
    coefficients = _LOESS(ts_train, frac=FRAC)
    domaine_freq = np.fft.fft(signal_no_tendance)
    f = np.fft.fftfreq(len(data))
    indices_tries = _tri_indice_freq(f)
    horizon = len(test)
    # nb_total_test = 200
    # evaluation parameter nb_harm
    # _optimisation_parameters()
    harm_div = 10
    nb_harm = int(len(data)/harm_div)
    #nb_harm = 50
    print('Coef pred tendance')
    print(coef_tendance)
    signal_pred = _prediction(horizon=horizon, nb_harm=nb_harm, domaine_freq=domaine_freq,
                               f=f, indices_tries=indices_tries, coef_tendance=coefficients)
    signal_pred_train = signal_pred[0:TRAIN_SIZE + 1]
    print('Signal restauré')
    print(signal_pred)
    print(len(signal_pred))
    print('Coefficents Loess')
    print(coefficients)
    last_coef = coefficients.values[-1]
    #signal_pred = _comb_signal(signal_pred, coefficients)
    signal_pred = _comb_signal(signal_pred, last_coef)

    signal_pred_test = signal_pred[TRAIN_SIZE:len(data)]

    # Preparation Plot
    mean_signal = int(sum(signal_pred)) / len(signal_pred)
    std = np.std(signal_pred)
    x_pred = [z for z in range(TRAIN_SIZE, int(len(data)))]
    born_sup = signal_pred_test + (mean_signal - 1.96 * std) / np.sqrt(len(test))
    born_inf = signal_pred_test - (mean_signal - 1.96 * std) / np.sqrt(len(test))

    signal_pred_train = pd.Series(index=ts_train.index, data=signal_pred_train)
    signal_pred_test = pd.Series(index=ts_test.index, data=signal_pred_test)

############

    # Plotting results
    trace0 = go.Scatter(
        x=data.index,
        y=data.Data,
        mode='markers',
        name='TS'
    )
    trace1 = go.Scatter(
        x=ts_train.index,
        y=signal_pred_train,
        mode='lines+markers',
        name='Model Fourier'
    )
    trace2 = go.Scatter(
        x=ts_test.index,
        y=signal_pred_test,
        mode='lines+markers',
        name='Predictions Fourier & Loess'
    )
    trace3 = go.Scatter(
        x=ts_test.index,
        y=born_sup,
        line=dict(
            color='gray',
            dash='dot')
    )
    trace4 = go.Scatter(
        x=ts_test.index,
        y=born_inf,
        line=dict(
            color='gray',
            dash='dot'
        )
    )
    trace5 = go.Scatter(
        x=ts_test.index,
        y=born_sup,
        fill='tonexty',
        mode='none'
    )
    layout = {
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
    }
    # layout = dict(
    #     title='Time Series & predictions',
    #     xaxis=dict(
    #         rangeslider=dict(
    #             visible=True
    #         ),
    #         type='date'
    #     )
    # )
    donnees = [trace0, trace1, trace2, trace3, trace4, trace5]
    fig = dict(data=donnees, layout=layout)
    py.plot(fig, filename='TS_FFT_Loess.html')

    # Estimation error on predictions
    _error(test, signal_pred_test)

if __name__ == '__main__':
    main()

    # plt.plot(ts, label='Production')
    # plt.plot(signal_pred_train, label='Model')
    # plt.plot(signal_pred_test, label='Prediction test')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(x_pred, test, label='Production test')
    # plt.plot(x_pred, signal_pred_test, label='Prediction signal')
    # plt.fill_between(x_pred, born_sup, born_inf, color='gray')
    # plt.xlabel("Jours")
    # plt.legend()
    # plt.show()