import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import Holt
from statsmodels.tsa.stattools import adfuller
from math import sqrt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

#plt.rcParams['figure.figsize']=(20,10)
# plt.style.use('ggplot')
# plt.ylim(0,10)


"""
def moving_average(signal, period):
  if len(signal)<=period:
    return np.nan
  result = signal[0:period].mean()
  for i in range(period+1, len(signal)):
    result += (signal[i] - signal[i-period]) / period
  return result
def moving_average(signal, period):
  result = np.nan
  if period>=len(signal):
    return result
  result = sum(signal[0:period])
  for i in range(period, len(signal)):
    result += signal[i] - signal[i - period]
  return result / period

  """


def moving_average(signal, period):
  if period>=len(signal):
    return np.nan
  return sum(signal[len(signal) - period:len(signal)])*1./period


def test_stationarity(timeseries):
  #Determing rolling statistics
  rolmean = pd.Series(timeseries).rolling(window=12).mean()
  rolstd = pd.Series(timeseries).rolling(window=12).std()
  #Plot rolling statistics:
  orig = plt.plot(timeseries, color='blue', label='Original')
  mean = plt.plot(rolmean, color='red', label='Rolling Mean')
  std = plt.plot(rolstd, color='black', label='Rolling Std')
  plt.legend(loc='best')
  plt.title('Rolling Mean & Standard Deviation')
  plt.show(block=False)
  #Perform Dickey-Fuller test:
  print('Results of Dickey-Fuller Test:')
  array = np.asarray(timeseries, dtype='float')
  np.nan_to_num(array, copy=False)
  dftest = adfuller(array, autolag='AIC')
  dfoutput = pd.Series(
      dftest[0:4],
      index=[
          'Test Statistic', 'p-value', '#Lags Used',
          'Number of Observations Used'
      ])
  for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value
  print(dfoutput)


def estimate_holt(data, alpha=0.2, slope=0.1, trend='add', fc=1):
  numbers = np.asarray(data, dtype='float')
  model = Holt(numbers)
  fit = model.fit(alpha, slope, trend)
  estimate = fit.forecast(fc)[-1]
  return estimate


def est_mv_avg(series, windowsize, num):
  estimated_values = []
  while (num > 0):
    cur_est = moving_average(series, windowsize)
    estimated_values.append(cur_est)
    series = np.append(series, [cur_est])
    num -= 1
  return estimated_values


def predictLastTwo(dataset):
  # Take out 15:57 and 15:58
  train = dataset[:-2]
  # Real values for 15:57 and 15:58
  true = dataset[-2:]

  # Estimate using Holt
  holt_est_1 = estimate_holt(
      train, alpha=0.2, slope=0.1, trend='add',
      fc=1)  # Holt Forecast 15:57  -> 6.026090018792743
  holt_est_2 = estimate_holt(
      train, alpha=0.2, slope=0.1, trend='add',
      fc=2)  # Holt Forecast 15:58  -> 6.0264529288475766
  print(true, holt_est_1, holt_est_2)
  holt_error = sqrt(mean_squared_error(true, [holt_est_1, holt_est_2]))
  print('Holt Estimations and Error:', holt_est_1, holt_est_2, holt_error)


def predictNextTwo(dataset):
  # Real values for 15:57 and 15:58
  true = dataset[-2:]
  # Estimate using Moving Avg.
  est1559, est1600 = est_mv_avg(dataset, 60, 2)
  mv_avg_error = sqrt(mean_squared_error(true, [est1559, est1600]))
  print('Moving Avg. Estimations and Error:', est1559, est1600, mv_avg_error)

def predictTomorrow4(dataset):
  estMay7_1600 = est_mv_avg(dataset, 60,
                            1442)[-1]  # 6.020135518518531
  print('May 7th, 16:00:', estMay7_1600)

def doPart1(dataset):
  #test_stationarity(trading['Open'])
  predictLastTwo(dataset['Open'].values)
  predictNextTwo(dataset['Open'].values)
  predictTomorrow4(dataset['Open'].values)


def doPart2(dataset):
  test_stationarity(trading['Open'])
  dataset = dataset[:-2]
  true_p2 = dataset[-2:]

  est1559_p2, est1600_p2 = est_mv_avg(dataset['Open'].values, 60, 2)
  mv_avg_error_p2 = sqrt(
      mean_squared_error(true_p2['Open'], [est1559_p2, est1600_p2]))
  print('Moving Avg. Estimations and Error: ', est1559_p2, est1600_p2,
        mv_avg_error_p2)


############## # PART 1 #############
trading = pd.read_csv('HW03_USD_TRY_Trading.txt', sep="\t")# parse_dates=[['Day','Time']], dayfirst=True)
doPart1(trading)

trading = trading[trading['Volume'] != 0]
trading = trading.dropna()

doPart1(trading)

##############
# PART 2
#############

trading = pd.read_csv('HW03_USD_TRY_Trading.txt', sep="\t")# parse_dates=[['Day','Time']], dayfirst=True)
trading = (trading[trading['Time'].str.contains(":(57|58)")])

doPart2(trading)
doPart2(trading[trading['Volume'] != 0].dropna())


#############
# PART 3
############

def performPart3Checks(dataset):
  test_stationarity(dataset)
  predictLastTwo(dataset)
  predictNextTwo(dataset)
  predictTomorrow4(dataset)

def doPart3(trading):
  # Simple moving average
  moving_avg_trading = trading['Open'].rolling(window=60).mean().dropna()
  plt.plot(moving_avg_trading)
  performPart3Checks(moving_avg_trading)

  # Linear weighted average
  moving_avg_trading = [np.nan] * len(trading['Open'])
  for start in range(len(trading['Open']) - 60):
    w = trading['Volume'][start:start + 60].values
    # Replace missing volumes with 0
    for i in range(len(w)):
      if np.isnan(w[i]):
        w[i] = 0
    if sum(w) == 0:
      w = [1. / len(w)] * len(w)
    moving_avg_trading[start + 60] = np.average(
        trading['Open'][start:start + 60], weights=w)
  test_stationarity(moving_avg_trading)
  plt.plot(moving_avg_trading)
  performPart3Checks(moving_avg_trading[60:])

  # Pick random
  for start in range(0, len(trading['Open']), 60):
    elem = np.random.choice(trading['Open'][start:start + 60])
    for i in range(60):
      if start + i >= len(moving_avg_trading):
        break
      moving_avg_trading[start + i] = elem
  test_stationarity(moving_avg_trading)
  plt.plot(moving_avg_trading)
  performPart3Checks(moving_avg_trading)

trading = pd.read_csv(
    'HW03_USD_TRY_Trading.txt',
    sep='\t')  # parse_dates=[['Day','Time']], dayfirst=True)
doPart3(trading)
