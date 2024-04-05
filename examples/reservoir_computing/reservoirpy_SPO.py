"""Example of back-tests with predictions obtained using reservoirpy.

    To run this, you need to install ``reservoirpy`` and ``hyperopt``.
"""

import os
from os import path
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from reservoirpy.nodes import Reservoir, Ridge, FORCE
from reservoirpy.observables import rmse, rsquare, nrmse
from reservoirpy.datasets import to_forecasting
from sklearn.preprocessing import MinMaxScaler
from reservoirpy.hyper import research


import cvxportfolio as cvx

# set parameters for reservoir computing
hyperopt_config = {
    "exp": f"hyperopt4", # the experimentation name
    "hp_max_evals": 20,             # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",           # the method used by hyperopt to choose those sets (see below)
    "seed": 40,                      # the random state seed, to ensure reproducibility
    "instances_per_trial": 3,        # how many random ESN will be tried with each sets of parameters
    "hp_space": {                    # what are the ranges of parameters explored
        "units": ["choice", 10, 15, 25, 35, 60, 70],             # the number of neurons
        "spectral_radius": ["loguniform", 1e-2, 10],   # the spectral radius is log-uniformly distributed between 1e-2 and 10
        "leak_rate": ["loguniform", 1e-3, 1],  # the leaking rate is log-uniformly distributed between from 1e-3 to 1
        "input_scaling": ["choice", 0.9],           # the input scaling is fixed
        "ridge": ["choice", 1e-7],        # the regularization parameter is fixed
        "seed": ["choice", 123]          # random seed for the ESN initialization
    }
}

# save the configuration in a JSON file
# each file will begin with a number corresponding to the current experimentation run number.
with open(f"examples/reservoir_computing/hyper_param_search/{hyperopt_config['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config, f)

# set parameters for getting data
data_param = {"stocks": ['GM', 'GE', 'AAPL', 'UBER', 'ZM', 'META', 'AMZN', 'TSLA', 'TTWO', 'QCOM'], "date_from": '2019-05-10', "date_to": '2024-03-01'}




def get_data(stocks_list, date_from = data_param['date_from'], date_to = data_param['date_to']):
    # Get a data frame with stocks returns, VIX and interest rate to use to forecast returns
    df = cvx.DownloadedMarketData(stocks_list).prices
    # stocks_df.dropna(inplace = True)
    dates = df.index[(df.index > date_from) & (df.index <= date_to)]
    df = df.loc[dates]
    df.dropna(inplace = True)
    # volumes = cvx.DownloadedMarketData(stocks_list).volumes
    # volumes = volumes.add_suffix('_vol')
    # df = df.merge(volumes, left_index = True, right_index = True, how = 'left')
    df.index = pd.to_datetime(df.index.date)
    VIX = pd.DataFrame({'VIX':cvx.YahooFinance('^VIX').data.open})
    VIX.index = pd.to_datetime(VIX.index.date)
    # rate = pd.DataFrame({'rate': cvx.DownloadedMarketData(['GE']).returns.USDOLLAR})
    # rate.index = pd.to_datetime(rate.index.date)
    df = df.merge(VIX, left_index = True, right_index = True, how = 'left')
    # df = df.merge(rate, left_index = True, right_index = True, how = 'left')
    return dates, df

""" def get_data(stocks_list, date_from = data_param['date_from'], date_to = data_param['date_to']):
    # Get a data frame with stocks returns, VIX and interest rate to use to forecast returns
    df = cvx.DownloadedMarketData(stocks_list).returns
    # stocks_df.dropna(inplace = True)
    dates = df.index[(df.index > date_from) & (df.index <= date_to)]
    df = df.loc[dates]
    df.dropna(inplace = True)
    # volumes = cvx.DownloadedMarketData(stocks_list).volumes
    # volumes = volumes.add_suffix('_vol')
    # df = df.merge(volumes, left_index = True, right_index = True, how = 'left')
    df.index = pd.to_datetime(df.index.date)
    df.rename(columns={"USDOLLAR": "rate"}, inplace = True)
    VIX = pd.DataFrame({'VIX':cvx.YahooFinance('^VIX').data.open})
    VIX.index = pd.to_datetime(VIX.index.date)
    df = df.merge(VIX, left_index = True, right_index = True, how = 'left')
    print(df)
    return dates, df """

def get_test_timeindex(dates, y_test):
    """Get Datetimeindex for test data"""
    return dates[-y_test.shape[0]:]

# define objective function, based on: https://reservoirpy.readthedocs.io/en/latest/user_guide/hyper.html

def objective(data_param, config, *, input_scaling, units, spectral_radius, leak_rate, ridge, seed):

    dates, data = get_data(data_param['stocks'])

    print("Start date:", str(dates[0].date()))
    # Split a timeseries for forecasting tasks.
    X = np.array(data)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = to_forecasting(X, forecast= 1, test_size = 0.35)
    test_index = get_test_timeindex(dates, y_test)
    dataset = ((X_train, y_train[:,:len(data_param['stocks'])]), (X_test, y_test[:,:len(data_param['stocks'])]))

    train_data, test_data = dataset
    X_train, y_train = train_data
    X_test, y_test = test_data

    instances = config["instances_per_trial"]

    variable_seed = seed

    losses = []; information_ratio = []
    for n in range(instances):
        # Build your model given the input parameters
        reservoir = Reservoir(units,
                              sr=spectral_radius,
                              lr=leak_rate,
                              input_scaling = input_scaling,
                              seed=variable_seed)

        readout = Ridge(ridge=ridge)

        model = reservoir >> readout


        # Train your model and test your model.
        predictions = model.fit(X_train, y_train) \
                           .run(X_test)
        
        scaler = MinMaxScaler()
        X = np.array(data)
        X = scaler.fit_transform(X[:,:len(data_param['stocks'])])
        print(predictions)
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test[:, :len(data_param['stocks'])])

        print("RMSE:", rmse(y_test, predictions), "R^2 score:", rsquare(y_test, predictions))
        # predicted_df = pd.DataFrame(data = predictions, index = test_index, columns = list(data.columns[:len(data_param['stocks'])]))
        predicted_df = pd.DataFrame(data = predictions, columns = list(data.columns[:len(data_param['stocks'])]))
        print(predicted_df)
        predicted_df = predicted_df.pct_change()[1:]
        predicted_df.index = test_index[:-1]
        
        gamma = 5       # risk aversion parameter 
        kappa = 0.05    # covariance forecast error risk parameter

        objective1 = cvx.ReturnsForecast(r_hat = predicted_df) - gamma * (
            cvx.FullCovariance() + kappa * cvx.RiskForecastError()
        ) - cvx.StocksTransactionCost() 

        # - 0.1 * cvx.ReturnsForecastError(cvx.forecast.HistoricalStandardDeviation)

        constraints = [cvx.LongOnly(), cvx.LeverageLimit(1), cvx.MaxWeights(0.2)]
        # , benchmark = cvx.Uniform()

        policy1 = cvx.SinglePeriodOptimization(objective1, constraints)
        simulator = cvx.StockMarketSimulator(data_param['stocks'])

        test_dates = predicted_df.index
        print(str(test_dates[0].date()))
        result = simulator.backtest(policy1, start_time = str(test_dates[0].date()), end_time = '2024-03-01')

     

        loss = - result.sharpe_ratio


        # Change the seed between instances
        variable_seed += 1

        losses.append(loss)
        information_ratio.append(result.information_ratio)


    # Return a dictionary of metrics. The 'loss' key is mandatory when
    # using hyperopt.
    return {'loss': np.mean(losses),
            'information ratio': np.mean(information_ratio),
            "RMSE": rmse(y_test, predictions),
            "R^2 score": rsquare(y_test, predictions)}

def get_best_params(result_path):
    report_path = path.join(result_path, "results")
    results = []
    for file in os.listdir(report_path):
        if path.isfile(path.join(report_path, file)):
            with open(path.join(report_path, file), "r") as f:
                results.append(json.load(f))
    id = np.argmin([r["returned_dict"]['loss'] for r in results])
    return results[id]['current_params']

# from reservoirpy
def plot_results(y_pred, y_test, sample=500):

    fig = plt.figure(figsize=(15, 7))
    plt.subplot(211)
    plt.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN prediction")
    plt.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True value")
    plt.plot(np.abs(y_test[:sample] - y_pred[:sample]), label="Absolute deviation")

    plt.legend()
    plt.show(block = False)

def get_predictions(data_param, hyperopt_config, forecast = 1, instances = 10, test_size = 0.2, hyper_search = True, online = True, seed = 123):
    """Get predictions."""
    # get data 
    dates, data = get_data(data_param['stocks'])

    print("Start date:", str(dates[0].date()))
    # Split a timeseries for forecasting tasks.
    X = np.array(data)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = to_forecasting(X, forecast= forecast, test_size = test_size)
    test_index = get_test_timeindex(dates, y_test)
    dataset = ((X_train, y_train[:,:len(data_param['stocks'])]), (X_test, y_test[:,:len(data_param['stocks'])]))

    train_data, test_data = dataset
    X_train, y_train = train_data
    X_test, y_test = test_data

    if hyper_search:
        # dataset = ((X_train, y_train[:,:len(data_param['stocks'])]), (X_test, y_test[:,:len(data_param['stocks'])]))
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, "examples/reservoir_computing/hyper_param_search")
        best = research(objective, data_param, f"examples/reservoir_computing/hyper_param_search/{hyperopt_config['exp']}.config.json", final_directory)

    param = get_best_params(hyperopt_config["exp"])
    print(param)

    predictions = []
    instances = hyperopt_config["instances_per_trial"]
    variable_seed = seed
    for i in range(instances):
        reservoir = Reservoir(units = param['units'], input_scaling=param['input_scaling'], sr=param['spectral_radius'],
                      lr=param['leak_rate'], seed=param['seed'])
        
        # if online:
            
        #     readout = FORCE(alpha = param['ridge'])

        #     esn_online = reservoir >> readout

        #     # train and test the model
        #     # esn_online.train(X_train, y_train[:, :len(data_param['stocks'])])
        #     esn_online.train(X_train, y_train)
        #     pred = esn_online.run(X_test)
        # else:
        readout = Ridge(ridge = param['ridge'])
        model = reservoir >> readout


        # Train your model and test your model.
        pred = model.fit(X_train, y_train) \
                           .run(X_test)
        
        predictions.append(pred)
        # Change the seed between instances
        variable_seed += 1

    predictions = np.mean(predictions, axis = 0)

    dates, data = get_data(data_param['stocks'])
    scaler = MinMaxScaler()
    X = np.array(data)
    print(predictions)
    X = scaler.fit_transform(X[:,:len(data_param['stocks'])])
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test[:, :len(data_param['stocks'])])
    # y_test = scaler.inverse_transform(y_test)
    # print example RMSE and R^2 score
    print("RMSE:", rmse(y_test[:, 1], predictions[:,1]), "R^2 score:", rsquare(y_test[:, 1], predictions[:,1]))
    print("RMSE:", rmse(y_test[:, 0], predictions[:,0]), "R^2 score:", rsquare(y_test[:, 0], predictions[:,0]))
    print("RMSE:", rmse(y_test, predictions), "R^2 score:", rsquare(y_test, predictions))
    plot_results(predictions[:,1], y_test[:,1], sample=400)
    # plot_results(predictions[:,0], y_test[:,0], sample=400)
    predicted_df = pd.DataFrame(data = predictions, columns = list(data.columns[:len(data_param['stocks'])]))
    # predicted_df = pd.DataFrame(data = predictions, index = test_index, columns = list(data.columns[:len(data_param['stocks'])]))
    print(predicted_df)
    # predicted_df.drop(columns=['VIX', 'rate'], inplace = True)
    # actual_df = pd.DataFrame(data = y_test, index = test_index, columns = list(data.columns[:len(data_param['stocks'])]))
    actual_df = pd.DataFrame(data = y_test, columns = list(data.columns[:len(data_param['stocks'])]))
    # actual_df.drop(columns=['VIX', 'rate'], inplace = True)
    predicted_df = predicted_df.pct_change()[1:]
    actual_df = actual_df.pct_change()[1:]
    predicted_df.index = test_index[:-1]
    actual_df.index = test_index[:-1]

    return predicted_df, actual_df

# set up the reservoir and get forecasted returns 1 day ahead
predicted, actual = get_predictions(data_param, hyperopt_config, forecast = 1, instances = 10, test_size = 0.35, hyper_search = True, online = True, seed = 123)

# Single Period Optimization using predictions

# parameters for SPO
gamma = 5       # risk aversion parameter (Chapter 4.2)
kappa = 0.05    # covariance forecast error risk parameter (Chapter 4.3)

objective1 = cvx.ReturnsForecast(r_hat = predicted) - gamma * (
    cvx.FullCovariance() + kappa * cvx.RiskForecastError()
) - cvx.StocksTransactionCost() 

# - 0.1 * cvx.ReturnsForecastError(cvx.forecast.HistoricalStandardDeviation)

constraints = [cvx.LongOnly(), cvx.LeverageLimit(1), cvx.MaxWeights(0.2)]
# , benchmark = cvx.Uniform()
policy1 = cvx.SinglePeriodOptimization(objective1, constraints)

objective2 = cvx.ReturnsForecast() - gamma * (
    cvx.FullCovariance() + kappa * cvx.RiskForecastError()
) - cvx.StocksTransactionCost()


policy2 = cvx.SinglePeriodOptimization(objective2, constraints)

simulator = cvx.StockMarketSimulator(
    data_param['stocks'])

# result = simulator.backtest(policy, start_time = str(dates[0].date()), end_time = '2024-03-01')
test_dates = predicted.index
print(str(test_dates[0].date()))
results = simulator.backtest_many([policy1, policy2, cvx.Uniform()], start_time = str(test_dates[0].date()), end_time = '2024-03-01')

results[0].plot()
results[1].plot()
results[2].plot()

# print statistics result of the backtest
print("\n# SINGLE-PERIOD OPTIMIZATION RESERVOIRPY:\n", results[0])
print("\n# SINGLE-PERIOD OPTIMIZATION DEFAULT:\n", results[1])
print("\n# UNIFORM ALLOCATION:\n", results[2])

print("Results:")

print("SPO reservoirpy profit:", results[0].profit)
print("SPO reservoirpy sharpe ratio:", results[0].sharpe_ratio)
print("SPO reservoirpy information ratio:", results[0].information_ratio)

print("SPO default profit:", results[1].profit)
print("SPO default sharpe ratio:", results[1].sharpe_ratio)
print("SPO default information ratio:", results[1].information_ratio)

print("Uniform allocation profit:", results[2].profit)
print("Uniform allocation sharpe ratio:", results[2].sharpe_ratio)
print("Uniform allocation information ratio:", results[2].information_ratio)

plt.show()