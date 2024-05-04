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

# set parameters for getting data
data_param = {"stocks": ['AAPL', 'ABNB', 'ADBE', 'AMZN', 'ANSS', 'ASML', 'CDW', 'CEG', 'CHTR', 'CTAS', 'CTSH', 'DASH','FTNT', 'GEHC', 'GFS','INTU', 'ISRG', 'KDP','MDLZ', 'MELI', 'META','NXPI', 'ODFL', 'ON','QCOM', 'REGN', 'ROP','TSLA', 'TTD', 'TTWO'], "date_from": '2019-05-10', "date_to": '2024-05-01'}
# define simulator
simulator = cvx.StockMarketSimulator(data_param['stocks'])
# define constraints
constraints = [cvx.LongOnly(), cvx.LeverageLimit(1), cvx.MaxWeights(0.1)]
# set parameters
hyperopt_config = {
    "exp": f"hyperopt_mult_01_returns", # the experimentation name
    "hp_max_evals": 100,             # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",           # the method used by hyperopt to choose those sets (random or tpe)
    "seed": 40,                      # the random state seed, to ensure reproducibility
    "instances_per_trial": 3,        # how many random ESN will be tried with each sets of parameters
    "stocks": data_param['stocks'],
    "H": 3, # horizon 
    "hp_space": {                    # what are the ranges of parameters explored
        "units": ["choice", 5, 10, 15, 25, 35, 60, 70],             # the number of neurons
        "spectral_radius": ["loguniform", 1e-2, 10],   # the spectral radius is log-uniformly distributed between 1e-2 and 10
        "leak_rate": ["loguniform", 1e-3, 1],  # the leaking rate is log-uniformly distributed between from 1e-3 to 1
        "input_scaling": ["choice", 0.9],           # the input scaling is fixed
        "ridge": ["choice", 1e-7],        # the regularization parameter is fixed
        "seed": ["choice", 123],         # random seed for the ESN initialization
        "gamma_risk": ["choice", 0.5, 1, 5, 10, 25, 50], # risk aversion parameter 
        "gamma_trade": ["choice", 0.5, 1, 5, 10, 25, 50], # trading risk aversion factor
    #   "gamma_hold": ["choice", 0.5, 1, 5, 10, 25, 50], # holdings aversion parameter
        "kappa": ["choice", 0.05, 0.1, 0.5]  # covariance forecast error risk parameter
    }
}

# save the configuration in a JSON file
# each file will begin with a number corresponding to the current experimentation run number.
with open(f"examples/reservoir_computing/hyper_param_search/{hyperopt_config['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config, f)

# get data
def get_data(stocks_list, date_from = data_param['date_from'], date_to = data_param['date_to']):
    # Get a data frame with stocks returns, VIX and interest rate to use to forecast returns
    df = cvx.DownloadedMarketData(stocks_list).returns
    df.dropna(inplace = True)
    dates = df.index[(df.index > date_from) & (df.index <= date_to)]
    df = df.loc[dates]
    # volumes = cvx.DownloadedMarketData(stocks_list).volumes
    # volumes = volumes.add_suffix('_vol')
    # df = df.merge(volumes, left_index = True, right_index = True, how = 'left')
    df.index = pd.to_datetime(df.index.date)
    df.rename(columns={"USDOLLAR": "rate"}, inplace = True)
    VIX = pd.DataFrame({'VIX':cvx.YahooFinance('^VIX').data.open})
    VIX.index = pd.to_datetime(VIX.index.date)
    df = df.merge(VIX, left_index = True, right_index = True, how = 'left')
    df.index = dates
    print(df)
    return df

def get_test_timeindex(dates, y_test):
    """Get Datetimeindex for test data"""
    return dates[-y_test.shape[0]:]

# define objective function, based on: https://reservoirpy.readthedocs.io/en/latest/user_guide/hyper.html

def objective(data, config, *, input_scaling, units, spectral_radius, leak_rate, ridge, seed, gamma_risk, gamma_trade, kappa):


    print("Start date:", str(data.index[0].date()))
    # Split a timeseries for forecasting tasks.
    X = np.array(data)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    input_data = {}

    for i in range(config["H"]):
        input_data[i+1] = list(to_forecasting(X, forecast= i+1, test_size = 0.35))
        input_data[i+1][2] = input_data[i+1][2][:,:len(config['stocks'])]
        input_data[i+1][3] = input_data[i+1][3][:,:len(config['stocks'])]


    scaler = MinMaxScaler()
    X = np.array(data)
    X = scaler.fit_transform(X[:,:len(data_param['stocks'])])

    for i in range(config["H"]):
        input_data[i+1][3] = scaler.inverse_transform(input_data[i+1][3])
        # input_data[i+1][3] = np.exp(input_data[i+1][3]) # use if predicting prices


    test_index = get_test_timeindex(data.index, input_data[i+1][3])
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

        predictions = {}
        # Train your model and test your model.
        for i in range(config["H"]):
            model = reservoir >> readout
            predictions[i+1] = model.fit(input_data[i+1][0], input_data[i+1][2]) \
                           .run(input_data[i+1][1])
            predictions[i+1] = scaler.inverse_transform(predictions[i+1])
            # predictions[i+1] = np.exp(predictions[i+1])
            print("RMSE:", rmse(input_data[i+1][3], predictions[i+1]), "R^2 score:", rsquare(input_data[i+1][3], predictions[i+1]))
            predictions[i+1] = pd.DataFrame(data = predictions[i+1], index = test_index, columns = list(data.columns[:len(data_param['stocks'])])) # use if predicting returns
            # predictions[i+1] = pd.DataFrame(data = predictions[i+1], columns = list(data.columns[:len(config['stocks'])])) # use if predicting prices
            # predictions[i+1] = predictions[i+1].pct_change()[1:] # use if predicting prices
            # predictions[i+1].index = test_index[:-1] # use if predicting prices

        print(predictions[1])

        
        objective = []

        for i in range(config["H"]):
            objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - gamma_risk * (
            cvx.FullCovariance() + kappa * cvx.RiskForecastError()
        ) - gamma_trade * cvx.StocksTransactionCost())


        # - 0.1 * cvx.ReturnsForecastError(cvx.forecast.HistoricalStandardDeviation)

        # , benchmark = cvx.Uniform()

        policy = cvx.MultiPeriodOptimization(objective, [constraints] * config["H"])

        test_dates = predictions[1].index
        print(str(test_dates[0].date()))
        result = simulator.backtest(policy, start_time = str(test_dates[0].date()), end_time = data_param['date_to'])

     

        loss = - result.sharpe_ratio


        # Change the seed between instances
        variable_seed += 1

        losses.append(loss)
        information_ratio.append(result.information_ratio)


    # Return a dictionary of metrics. The 'loss' key is mandatory when
    # using hyperopt.
    return {'loss': np.mean(losses),
            'information ratio': np.mean(information_ratio),
            "RMSE": rmse(predictions[1], predictions[1]),
            "R^2 score": rsquare(predictions[1], predictions[1])}

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
    data = get_data(data_param['stocks'])
    # data.iloc[:,:len(data_param['stocks'])] = np.log(data.iloc[:,:len(data_param['stocks'])])

    if hyper_search:
        # dataset = ((X_train, y_train[:,:len(data_param['stocks'])]), (X_test, y_test[:,:len(data_param['stocks'])]))
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, "examples/reservoir_computing/hyper_param_search")
        best = research(objective, data, f"examples/reservoir_computing/hyper_param_search/{hyperopt_config['exp']}.config.json", final_directory)

    param = get_best_params(f"examples/reservoir_computing/hyper_param_search/{hyperopt_config['exp']}")
    print(param)

    print("Start date:", str(data.index[0].date()))
    # Split a timeseries for forecasting tasks.
    X = np.array(data)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    input_data = {}

    for i in range(hyperopt_config["H"]):
        input_data[i+1] = list(to_forecasting(X, forecast= i+1, test_size = 0.35))
        input_data[i+1][2] = input_data[i+1][2][:,:len(hyperopt_config['stocks'])]
        input_data[i+1][3] = input_data[i+1][3][:,:len(hyperopt_config['stocks'])]


    scaler = MinMaxScaler()
    X = np.array(data)
    X = scaler.fit_transform(X[:,:len(data_param['stocks'])])

    for i in range(hyperopt_config["H"]):
        input_data[i+1][3] = scaler.inverse_transform(input_data[i+1][3]) # y_test
        # input_data[i+1][3] = np.exp(input_data[i+1][3]) # use if predicting prices

    test_index = get_test_timeindex(data.index, input_data[i+1][3])

    predictions = {}

    instances = hyperopt_config["instances_per_trial"]
    
    for i in range(hyperopt_config["H"]):
        variable_seed = seed
        pred = []
        for j in range(instances):
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
            pred.append(model.fit(input_data[i+1][0], input_data[i+1][2]) \
                           .run(input_data[i+1][1]))
            # Change the seed between instances
            variable_seed += 1
            
        pred = np.mean(pred, axis = 0)
        predictions[i+1] = scaler.inverse_transform(pred)
        # predictions[i+1] = np.exp(predictions[i+1]) # use if predicting prices
        print("RMSE:", rmse(input_data[i+1][3], predictions[i+1]), "R^2 score:", rsquare(input_data[i+1][3], predictions[i+1])) 
        predictions[i+1] = pd.DataFrame(data = predictions[i+1], index = test_index, columns = list(data.columns[:len(data_param['stocks'])])) # use if predicting returns
        # predictions[i+1] = pd.DataFrame(data = predictions[i+1], columns = list(data.columns[:len(config['stocks'])])) # use if predicting prices
        # predictions[i+1] = predictions[i+1].pct_change()[1:] # use if predicting prices
        # predictions[i+1].index = test_index[:-1] # use if predicting prices
        
       # plot sample results
    plot_results(np.array(predictions[1].iloc[:,1]), input_data[1][3][:,1], sample=100)

    print(predictions[1])

    return predictions

# set up the reservoir and get forecasted returns 1 day ahead
predictions = get_predictions(data_param, hyperopt_config, forecast = hyperopt_config["H"], instances = 10, test_size = 0.35, hyper_search = True, online = True, seed = 123)

param = get_best_params(f"examples/reservoir_computing/hyper_param_search/{hyperopt_config['exp']}")

objective = []

for i in range(hyperopt_config["H"]):
    objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - param['gamma_risk'] * (
    cvx.FullCovariance() + param['kappa'] * cvx.RiskForecastError()
) - param['gamma_trade'] * cvx.StocksTransactionCost())
            
# - 0.1 * cvx.ReturnsForecastError(cvx.forecast.HistoricalStandardDeviation)

# , benchmark = cvx.Uniform()

policy = cvx.MultiPeriodOptimization(objective, [constraints] * 3)


test_dates = predictions[1].index
print(str(test_dates[0].date()))

# result = simulator.backtest(policy, start_time = str(test_dates[0].date()), end_time = data_param['date_to'])

results = simulator.backtest_many([policy, cvx.Uniform()], start_time = str(test_dates[0].date()), end_time = data_param['date_to'])

results[0].plot()
results[1].plot()

# print statistics result of the backtest
print("\n# MULTI-PERIOD OPTIMIZATION RESERVOIRPY:\n", results[0])
print("\n# UNIFORM ALLOCATION:\n", results[1])

print("Results:")

print("MPO reservoirpy profit:", results[0].profit)
print("MPO reservoirpy sharpe ratio:", results[0].sharpe_ratio)
print("MPO reservoirpy information ratio:", results[0].information_ratio)

print("Uniform allocation profit:", results[1].profit)
print("Uniform allocation sharpe ratio:", results[1].sharpe_ratio)
print("Uniform allocation information ratio:", results[1].information_ratio)

plt.show()