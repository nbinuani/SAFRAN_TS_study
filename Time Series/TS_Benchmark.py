# -*- coding utf-8 -*-
"""
MAJ : 31/10/18
S598658 Binuani Nicolas
"""

# % Librairies
from tkinter.filedialog import askopenfilename
import warnings
import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import interp1d
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from sklearn.metrics import mean_squared_error
import plotly.offline as py
import plotly.graph_objs as go
from scipy import interpolate
#######################################################################################################################
class Data_eng():
    """
    Class with all the basic functions for the drift monitoring tool
    """
    def _parameters_entries(self):
        """
        define the paramters for the study
        :return: filepath, nb_predictions, limit_sup, limit_inf, start, stop
        """
        filepath = askopenfilename(title="Choisir le fichier csv",
                                   filetypes=[('csv files', '.csv'), ('all files', '.*')])
        nb_predictions = int(input("Number of predictions day (int) :"))
        limit_sup = float(input("Limit sup of the signal (float) :"))
        limit_inf = float(input("Limit inf of the signal (float) :"))
        start = int(input("Signal start (int) :"))
        stop = int(input("Signal stop (int) :"))
        FRAC = float(input("FRAC - Loess (1/=>) : "))
        FRAC = 1/FRAC
        compute_vol = bool(input("Compute volatility SARIMA (True or False) :"))
        if compute_vol == None:
            compute_vol = False
        else:
            pass

        return filepath, nb_predictions, limit_sup, limit_inf, start, stop, FRAC, compute_vol

    def _loading_data(self, path, nb_predictions, start, stop):
        """
        load the TS & split it
        :param path:
        :param nb_predictions: day of predictions
        :param start: first day (int)
        :param stop: last day including the predictions (int)
        :return: ts_train, ts_test, ts, data, WD, TRAIN_SIZE
        """
        dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
        data = pd.read_csv(path, sep=',', parse_dates=['TimeIndex'], index_col='TimeIndex', date_parser=dateparse)
        data = data[start:stop]
        ts = data["Data"]
        SPLIT = 1 - (nb_predictions / len(ts))
        TRAIN_SIZE = int(len(ts) * SPLIT)
        WD = int(len(data) / 30)  # window for test stationnary
        ts_train, ts_test = ts[0:TRAIN_SIZE + 1], ts[TRAIN_SIZE:len(ts)]
        return ts_train, ts_test, ts, data, WD, TRAIN_SIZE, SPLIT

#######################################################################################################################
class TS_basic_func():
    """
    Class with all the basic functions for the drift monitoring tool
    """
    def ts_plot(y, lags=None, title=''):
        '''
        Compute & plot acf, pacf, histogram, and qq plot for a given time series
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

    def _test_stationarity(self, timeseries):
        """
        Apply a Dickey-Fuller test to verify the stationnarity of the TS
        :param timeseries: TS
        :return: stationnary => 1 or 0
        """
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
        return stationnary

    def stationnary_coef(self, data, type):
        """
        Create the coefficient differenciation variable
        :param data: dataframe of the TS
        :param type: type = what is the signal tested (for information and separate when they are too many tests)
        :return: diff's coefficient (0 or 1), the serie differenciated ts_diff
        """
        print('Test Stationanry '+str(type))
        stationary = self._test_stationarity(data)
        differenciation = 0
        ts_diff = data
        if stationary == False:
            differenciation = 1
            ts_diff -= ts_diff.shift(differenciation)
        return differenciation, ts_diff

    def mydrift(self, predictions, lim_sup, lim_inf):
        """
        The function to color the prediction and alert
        :param predictions:
        :param lim_sup:
        :param lim_inf:
        :return:
        """
        # color = ['red', 'orange', 'green']
        color = ['rgb(215, 11, 11)', 'rgb(240, 140, 0)', 'rgb(0, 204, 0)']
        nb_pts_critic = int(len(predictions) / 2)
        clign = color[2]
        if predictions[: nb_pts_critic].mean() > lim_sup or predictions[: nb_pts_critic].mean() < lim_inf:
            clign = color[0]
        elif predictions[nb_pts_critic:].mean() > lim_sup or predictions[nb_pts_critic:].mean() < lim_inf:
            clign = color[1]
        # print('Code couleur des avertissements :')
        # print('Rouge = premières valeurs de prédiction au dessus des limites = état critique')
        # print('Orange = dernières valeurs de prédiction au dessus des limites = avertissement')
        # print('Vert = RAS')
        # print(clign)
        return clign

    def error(self, original, create):
        error = mean_squared_error(original, create)
        print('MSE prediction : ' + str(error))
        return error

    def my_reglin(self, ts_test, model):
        LIN_TRAIN = len(ts_test)
        x = [l for l in range(len(model))]
        x_pred = x[len(x) - LIN_TRAIN:]
        model_ts = model[len(model) - LIN_TRAIN:]
        ## Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x=x_pred, y=model_ts.values)
        trend = []
        for w in range(len(x_pred)):
            trend.append(slope * x_pred[w] + intercept)
        return trend

#######################################################################################################################
class TS_SARIMA_variability():
    """
    Class for the ARIMA Model
    """
    def _evaluate_models(self, train, d, d_seasonal, WD):
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

    def _evaluate_garch_model(self, train, test, split_time):
        """
        Optimize Garch model
        :param train: train part of the signal
        :param test: test part of the signal
        :param split_time: point when you start to predict
        :return: variance volatility
        """
        warnings.filterwarnings("ignore")
        best_score = float(10000)
        best_p = (1)
        best_q = int(1)
        p_values = range(1, 3)
        q_values = range(1, 2)
        for p in p_values:
            for q in q_values:
                try:
                    am = arch_model(train, vol='Garch', p=p, q=q)
                    res = am.fit(last_obs=split_time)
                    aic = SARIMAXResults.aic(res)
                    if abs(aic) < best_score:
                        best_score, best_p, best_q = aic, p, q
                        # print('GARCH : p=%i  GARCH : q=%i  AIC=%.3f' % (p, q, aic))
                except:
                    continue
        print('Best config GARCH model p=%i q=%i AIC=%.3f' % (best_p, best_q, best_score))
        am = arch_model(train, vol='Garch', p=best_p, q=best_q)
        res = am.fit(last_obs=split_time)
        forecast_analytic = res.forecast(horizon=len(test), method='analytic')
        forecast_variance = forecast_analytic.variance[split_time:]
        forecast_volatility = forecast_analytic.residual_variance[split_time:]
        forecast_garch = forecast_variance.iloc[:, 0]
        forecast_vol = forecast_volatility.iloc[:, 0]
        return forecast_garch, forecast_vol

    def _predictions_final(self, pred_sarima, forecast_garch):
        """
        Final prediction
        :param pred_sarima: series
        :param forecast_garch: series
        :return: final prediction with volatility (Garch)
        """
        final_predictions = (pred_sarima + forecast_garch) / 2
        return final_predictions

    def test_model_sarima(self, ts_train, ts_test, d, d_seasonal, TRAIN_SIZE, WD, compute_vol):
        """
        test Sarima
        :param ts_train: time series train
        :param ts_test: time series test
        :param d: coefficient for the signal stationnarity
        :param d_seasonal: coefficient for the seasonal stationnarity
        :param TRAIN_SIZE: int
        :param WD: int
        :param compute_vol: bool
        :return: train part sarima + prediction sarima => time series
        """
        warnings.filterwarnings("ignore")
        order_arima, order_seasonal, cfg = self._evaluate_models(train=ts_train, d=d,
                                                            d_seasonal=d_seasonal, WD=WD)
        model = SARIMAX(ts_train, order=order_arima,seasonal_order=order_seasonal)
        md_fit = model.fit()
        ts_sarima = pd.Series(md_fit.fittedvalues)
        predictions_sarima = md_fit.predict(start=TRAIN_SIZE, end=TRAIN_SIZE + len(ts_test) -1)
        model_sarima = pd.Series(index=ts_train.index, data=ts_sarima)
        pred_sarima = pd.Series(index=ts_test.index, data=predictions_sarima)
        ts_SARIMA = pd.concat([model_sarima, pred_sarima], axis=0)
        ts_SARIMA.drop_duplicates(inplace=True)
        split_date = ts_train.index[-1]
        print(split_date)
        if compute_vol == True:
            forecast, volatility = self._evaluate_garch_model(train=ts_train, test=ts_test, split_time=split_date)
            pred_final = self._predictions_final(pred_sarima, forecast)
            return model_sarima, pred_sarima, pred_final, forecast
        else:
            return model_sarima, pred_sarima

#######################################################################################################################
class TS_Fourier():
    def _retirerTendance(self, data):
        """
        Remove the trend by a simple Linear reg and save the coefficients
        :param data: series of the initial signal
        :return:signal with no trend, the linear reg. of the trend (array)
        """
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
        slope = (len(data) * sum_xy - sum_x * sum_y) / (len(data) * sum_xx - sum_x * sum_x)
        for j in range(0, len(data)):
            data_no_trend[j] = data[j] - slope * j
        return data_no_trend, slope

    def _loess(self, data, frac):
        """
        Loess regression
        :param data: Series - data signal
        :param frac: fraction - <1
        :return: Series of the coefficient regression
        """
        lowess_ts = lowess(endog=data.values, exog=data.index, return_sorted=False, frac=frac)
        lowess_TS = pd.Series(index=data.index, data=lowess_ts)
        return lowess_TS

    def _tri_indice_freq(self, freqs):
        """

        :param freqs:
        :return:
        """
        indice = [i for i in range(0, int(len(freqs)))]
        tmpTab = pd.DataFrame({'freqs': freqs, 'indice': indice})
        tmpTab['abs'] = abs(tmpTab.freqs)
        tmpTab.sort_values(by='abs', inplace=True)
        return tmpTab.indice

    def _prediction(self, horizon, nb_harm, domaine_freq, f, indices_tries):
        """
        Prediction for Fourier signal
        :param horizon: number of prediction day - int
        :param nb_harm: number of sinusoides use to recompose the signal
        :param domaine_freq:
        :param f:
        :param indices_tries:
        :param coef_tendance:
        :return:
        """
        signal_restaure = [0 for i in range(0, (len(domaine_freq) + horizon - 1))]
        for k in range(0, 1 + nb_harm * 2):
            indice = indices_tries[k]
            amplitude = sqrt(pow(domaine_freq[indice].real, 2) + pow(domaine_freq[indice].imag, 2)) / len(domaine_freq)
            phase = atan2(domaine_freq[indice].imag, domaine_freq[indice].real)
            facteur = 2 * pi * f[indice]
            for j in range(0, (len(domaine_freq) + horizon - 1)):
                signal_restaure[j] += amplitude * (sin(facteur * j + phase))
        return signal_restaure

    def _comb_signal(self, signal_restaure, coef_tendance):
        """
        Sum the result of the regression & the Fourier
        :param signal_restaure: signal from Fourier - list
        :param coef_tendance: coef from loess regression - list
        :return: sum signal
        """
        for l in range(len(signal_restaure)):
            signal_restaure[l] = signal_restaure[l] + coef_tendance[l]
        return signal_restaure

    def fourier_results(self, ts_train, ts_test, FRAC, data_index):
        """

        :param ts_train:
        :param ts_test:
        :param FRAC: fraction of the loess regression - < 1
        :param data_index: All the index of the signal (data)
        :return: the train part & the prediction of Fourier method
        """
        coefficients_train = self._loess(ts_train, frac=FRAC)
        index_train = ts_train.index
        TS_train = pd.Series(index=index_train, data=coefficients_train.values)
        WINDOW = len(ts_test)
        coefficients_wd = coefficients_train[len(coefficients_train) - WINDOW:]
        coefficients_train.reset_index(drop=True, inplace=True)
        f = interp1d(coefficients_train.index, coefficients_train.values)
        first_coef = coefficients_train.values[1]
        last_coef = coefficients_train.values[-1]
        ## The delta is used to have a continuous signal (without a shift between the training & prediction)
        delta = last_coef - first_coef
        DELTA = 0.1
        x_new = [x for x in range(len(ts_test))]
        if (abs(delta) > DELTA):
            y_predictions = f(x_new) + delta
            print("ajout delta")
        else:
            y_predictions = f(ts_test.index)

        TS_test = pd.Series(index=ts_test.index, data=y_predictions)
        TS_loess = pd.concat([TS_train, TS_test], axis=0)

        # FFT
        domaine_freq = np.fft.fft(ts_train)
        f = np.fft.fftfreq(len(ts_train))
        indices_tries = self._tri_indice_freq(f)
        horizon = len(ts_test)
        harm_div = 10
        nb_harm = int(len(ts_train) / harm_div)
        signal_pred = self._prediction(horizon=horizon, nb_harm=nb_harm, domaine_freq=domaine_freq,
                                  f=f, indices_tries=indices_tries)

        # Final
        array_final = self._comb_signal(signal_pred, TS_loess)
        Four_final = pd.Series(index=data_index, data=array_final)
        Four_model = Four_final[0:len(ts_train)]
        predictions = Four_final[len(ts_train)-1:(len(ts_train)+len(ts_test))]

        return Four_final, predictions

#######################################################################################################################
class TS_AR():
    """
    Class of AR modelling
    """
    def AR(self, ts, SPLIT):
        """
        Make Autoregressive function
        :param ts: time series
        :param SPLIT: to separate Train part & Test part
        :return:plot of the prediction
        """
        # create lagged dataset
        dataframe = pd.concat([ts.shift(1), ts], axis=1)
        dataframe.columns = ['t-1', 't+1']
        # split into train and test sets
        X = dataframe.values
        train_size = int(len(X) * SPLIT)
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

        ts_train, ts_test = ts[0:train_size + 1], ts[train_size:len(ts)]
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
            predictions.append(-1 * pred_error)
            history.append(error)
            # print('predicted error=%f, expected error=%f' % (pred_error, error))
        # self._plot_ar(expected_error, predictions)
        self._plot_ar(expected_error, predictions)
        return

    def _plot_ar(self, exp_err, pred):
        """
        plot the prediction & the expected error
        :param exp_err: expected error
        :param pred: prediction
        :return: matplotlib plot
        """
        # plot predicted error
        plt.plot(exp_err)
        plt.plot(pred, color='red')
        plt.legend()
        plt.show()


#######################################################################################################################
class Visualisation():
    """
    Class to make the plotly results
    """
    def _born(self, predictions, ts_test):
        mean_pred = int(sum(predictions)) / len(predictions)
        std_pred = np.std(predictions)
        born_sup = predictions + (mean_pred + 1.96 * std_pred) / np.sqrt(len(ts_test))
        born_inf = predictions - (mean_pred - 1.96 * std_pred) / np.sqrt(len(ts_test))
        return born_sup, born_inf

    def _plot(self, data, model, predictions, color, trend_linear, limit_sup, limit_inf, ts_test, draw_vol, volatility, filename):
        first_element = data.index[0]
        last_element = data.index[-1]
        born_sup, born_inf = self._born(predictions, ts_test)

        #Plotting results
        trace0 = go.Scatter(
            x=data.index,
            y=data.Data,
            mode='markers',
            name='TS'
        )
        trace1 = go.Scatter(
            x=model.index,
            y=model,
            mode='lines',
            name='Train Set',
            marker=dict(
                color='rgb(11, 133, 215)'
            )
        )
        trace2 = go.Scatter(
            x=predictions.index,
            y=predictions,
            mode='markers',
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
                color=color)
        )
        trace7 = go.Scatter(
            x=predictions.index,
            y=volatility,
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
        if draw_vol == True:
            donnees = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7]
        else:
            donnees = [trace0, trace1, trace2, trace3, trace4, trace5, trace6]
        fig = dict(data=donnees, layout=layout)
        py.plot(fig, filename=filename)

#######################################################################################################################
class Create_TS():
    """
    Class to create a time serie as you want (modify the shift and the size etc...)
    """
    def TS_csv(self):
        Time_Series1 = np.random.laplace(scale=0.5, size=90)
        Time_Series2 = np.random.laplace(scale=0.5, size=30) + 1.5
        Time_Series3 = np.random.laplace(scale=0.5, size=30) + 0.7
        Time_Series4 = np.random.laplace(scale=0.5, size=60) + 1
        Time_Series5 = np.random.laplace(scale=0.5, size=30) + 0.9
        # for i in range(0,len(Time_Series)):
        #     if Time_Series[i] < 0:
        #         Time_Series[i]=0
        #     else:
        #         Time_Series[i] = Time_Series[i]
        TS1 = pd.Series(Time_Series1)
        TS2 = pd.Series(Time_Series2)
        TS3 = pd.Series(Time_Series3)
        TS4 = pd.Series(Time_Series4)
        TS5 = pd.Series(Time_Series5)
        Time_Series = pd.concat([TS1, TS3, TS5, TS4, TS2, TS2], axis=0)
        Time = pd.DatetimeIndex(Time_Series, start=2008)
        time = pd.datetime(year=2015, month=1, day=1)
        dates = pd.date_range(start='2015-01-01', periods=len(Time_Series), freq='D')
        data = {'TimeIndex': dates, 'Data': Time_Series}
        Data = pd.DataFrame(data=data)
        Data.set_index('TimeIndex', inplace=True)
        plt.scatter(x=Data.index, y=Data)
        plt.show()
        # print(Data)
        Data.to_csv('TS_created.csv', sep=',')
        return print('You create a time serie (be carefull with the name of the csv file)')

#######################################################################################################################
class Controller():
    def run(self):
        # Load DATA
        data_eng = Data_eng()
        functs = TS_basic_func()
        sarima = TS_SARIMA_variability()
        four = TS_Fourier()
        ar = TS_AR()
        viz = Visualisation()
        outut_TS = Create_TS()

        ## if you want to create a time serie
        # outut_TS.TS_csv()

        path, nb_predictions, limit_sup, limit_inf, start, stop, FRAC, bool_vol = data_eng._parameters_entries()
        ts_train, ts_test, ts, data, WD, TRAIN_SIZE, SPLIT = data_eng._loading_data(path, nb_predictions, start, stop)
        index = data.index
        index_test = ts_test.index

        # Decompostition
        decomposition = seasonal_decompose(ts, freq=WD)
        seasonal = decomposition.seasonal

        d, ts_train_diff = functs.stationnary_coef(ts_train, type='Time Series')
        ts_train_diff.fillna(ts_train_diff.mean(), inplace=True)
        d_seasonal, seasonal_diff = functs.stationnary_coef(seasonal, type='Seasonality')

        #SARIMA
        model_sarima, predictions_sarima, pred_final, volatility = sarima.test_model_sarima(ts_train=ts_train, ts_test=ts_test,
                                                                                            d=d, d_seasonal=d_seasonal,
                                                                                            TRAIN_SIZE=TRAIN_SIZE, WD=WD,
                                                                                            compute_vol=bool_vol)
        #Fourier
        frac = FRAC
        # frac = 1/5
        if (d == 1):
            model_four, predictions_four = four.fourier_results(ts_train=ts_train_diff, ts_test=ts_test, FRAC=frac, data_index=index)
        else:
            model_four, predictions_four = four.fourier_results(ts_train=ts_train, ts_test=ts_test, FRAC=frac, data_index=index)

        #AR
        # if (d == 1):
        #     ts_diff = ts - ts.shift(d)
        #     ts_diff.fillna(ts_diff.mean(), inplace=True)
        #     ar.AR(ts_diff, SPLIT=SPLIT)
        # else:
        #     ar.AR(ts, SPLIT=SPLIT)
        ts_test = pd.Series(index=index_test, data=ts_test.values)

        trend_sarima = functs.my_reglin(ts_test, model_sarima)
        sarima_color = functs.mydrift(predictions_sarima, limit_sup, limit_inf)
        trend_four = functs.my_reglin(ts_test, model_four)
        fourier_color = functs.mydrift(predictions_four, limit_sup, limit_inf)

        # Estimation error on predictions
        functs.error(ts_test, predictions_sarima)
        functs.error(ts_test, predictions_four)

        # Plot results
        viz._plot(data, model=model_sarima, predictions=predictions_sarima, trend_linear=trend_sarima, color=sarima_color,
                  limit_sup=limit_sup, limit_inf=limit_inf, ts_test=ts_test, filename='Sarima.html', draw_vol=bool_vol,
                  volatility=volatility)

        viz._plot(data, model=model_four, predictions=predictions_four, trend_linear=trend_four, color=fourier_color,
                  limit_sup=limit_sup, limit_inf=limit_inf, ts_test=ts_test, filename='Fourier.html', draw_vol=False,
                  volatility=None)


if __name__ == '__main__':
    c = Controller()
    c.run()