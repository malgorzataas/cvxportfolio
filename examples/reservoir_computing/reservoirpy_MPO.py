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
from sklearn.preprocessing import FunctionTransformer
from reservoirpy.hyper import research
from cvxportfolio.errors import PortfolioOptimizationError

import cvxportfolio as cvx

NDX100 = \
['AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMD', 'AMGN',
 'AMZN', 'ANSS', 'ASML', 'AVGO', 'AZN', 'BIIB', 'BKNG', 'BKR', 'CCEP', 'CDNS',
 'CDW', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSGP', 'CSX',
 'CTAS', 'CTSH', 'DASH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EXC', 'FANG', 'FAST',
 'FTNT', 'GEHC', 'GFS', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN', 'INTC',
 'INTU', 'ISRG', 'KDP', 'KHC', 'KLAC', 'LIN', 'LRCX', 'LULU', 'MAR', 'MCHP',
 'MDB', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA', 'MRVL', 'MSFT', 'MU', 'NFLX',
 'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP',
 'PYPL', 'QCOM', 'REGN', 'ROP', 'ROST', 'SBUX', 'SIRI', 'SNPS', 'TEAM', 'TMUS',
 'TSLA', 'TTD', 'TTWO', 'TXN', 'VRSK', 'VRTX', 'WBA', 'WBD', 'WDAY', 'XEL',
 'ZS']

# UNIVERSE = ['AAPL', 'ABNB', 'ADBE', 'AMZN', 'ANSS', 'ASML', 'CDW', 'CEG',
#             'CHTR', 'CTAS', 'CTSH', 'DASH','FTNT', 'GEHC', 'GFS','INTU',
#             'ISRG', 'KDP','MDLZ', 'MELI', 'META','NXPI', 'ODFL', 'ON',
#             'QCOM', 'REGN', 'ROP','TSLA', 'TTD', 'TTWO', 'TXN',
#             'VRSK', 'VRTX', 'WBA', 'WBD', 'WDAY', 'XEL']

UNIVERSE = ['AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMD', 'AMGN',
       'AMZN', 'ANSS', 'AZN', 'BIIB', 'BKNG', 'CDNS', 'CMCSA', 'COST', 'CPRT',
       'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'DLTR', 'EA', 'EXC', 'FAST',
       'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN', 'INTC', 'INTU', 'ISRG',
       'LIN', 'LRCX', 'MAR', 'MCHP', 'MNST', 'MRVL', 'MSFT', 'MU', 'NFLX',
       'NVDA', 'ODFL', 'ON', 'ORLY', 'PAYX', 'PCAR', 'PEP', 'QCOM', 'REGN',
       'ROP', 'ROST', 'SBUX', 'SIRI', 'SNPS', 'TTWO', 'TXN', 'VRTX', 'WBA',
       'XEL']
# set parameters for getting data
# "date_from": '2019-05-10', '2015-01-01'
data_param = {"stocks": NDX100, 'keep_stocks': False, "date_from": '2019-05-10', "date_to": '2024-05-01', 'H': 3, "long": True, "diagonal_cov": True, 'soft_constraints': True, 'train_set': 1} # H = planning horizon, keep_stocks: if True we'll keep all stocks listed and use only dates for which all stocks have data, otherwise stocks without data for specified dates will be dropped
# train_set = how many years to take for training, diagonal_cov = whether to use diagonalcovariance, long = long or longshort portfolio

# get data
def get_data(stocks_list, keep_stocks = data_param['keep_stocks'], date_from = data_param['date_from'], date_to = data_param['date_to']):
    # Get a data frame with stocks returns, VIX and interest rate to use to forecast returns
    df = cvx.DownloadedMarketData(stocks_list).returns
    if keep_stocks:
        df.dropna(inplace = True)
    dates = df.index[(df.index > date_from) & (df.index <= date_to)]
    df = df.loc[dates]
    if not keep_stocks:
        df.dropna(axis = 1, inplace = True)
        print(len(df.columns))
        print(df.columns)
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

data = get_data(data_param['stocks'])
# define simulator
simulator = cvx.StockMarketSimulator(list(data.columns[:-2]))
# add dollar neutral
# get objectives' parameters 
obj_params = {}
for i in range(data_param["H"]):
    obj_params['kappa_' + str(i+1)] = ["choice", 0, 0.05, 0.1, 0.5]  # covariance forecast error risk parameter
    obj_params['gamma_risk_' + str(i+1)] = ["choice", 0.5, 1, 5, 10, 25, 50]  # risk aversion parameter
    obj_params['gamma_trade_' + str(i+1)] = ["choice", 0.5, 1, 5, 10, 25, 50]  # trading risk aversion factor
    if not data_param['long']:
        obj_params['gamma_hold_' + str(i+1)] = ["choice", 0.5, 1, 5, 10, 25, 50] # holdings aversion parameter


# set parameters
hyperopt_config = {
    "hyper_search": True, # run hyper-parameter optimization or not
    "exp": f"hyperopt_mult_ndx100_long_diagonal_3_soft", # the experimentation name
    "hp_max_evals": 100,             # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",           # the method used by hyperopt to choose those sets ("random" or "tpe")
    "seed": 40,                      # the random state seed, to ensure reproducibility
    "instances_per_trial": 3,        # how many random ESN will be tried with each sets of parameters
    "hp_space": {                    # what are the ranges of parameters explored
        "units": ["choice", 5, 10, 15, 25, 35, 60, 70],             # the number of neurons
        "spectral_radius": ["loguniform", 1e-2, 10],   # the spectral radius is log-uniformly distributed between 1e-2 and 10
        "leak_rate": ["loguniform", 1e-3, 1],  # the leaking rate is log-uniformly distributed between from 1e-3 to 1
        "input_scaling": ["choice", 0.7, 0.9],           # the input scaling is fixed
        "ridge": ["choice", 1e-5, 1e-7, 1e-9],        # the regularization parameter is fixed
        "seed": ["choice", 123],         # random seed for the ESN initialization
        "weight": ["choice", 0.05, 0.1, 0.2], # weights constraint
        "leverage": ["choice", 1, 3, 5], # leverage limit
        "turnover": ["choice", 0.01, 0.05, 0.1, 0.3, 0.5], # turnover limit
        **obj_params
    }
}

print(hyperopt_config)

# save the configuration in a JSON file
# each file will begin with a number corresponding to the current experimentation run number.
with open(f"examples/reservoir_computing/hyper_param_search/{hyperopt_config['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config, f)

def get_test_timeindex(dates, y_test):
    """Get Datetimeindex for test data"""
    return dates[-y_test.shape[0]:]

# define objective function, based on: https://reservoirpy.readthedocs.io/en/latest/user_guide/hyper.html

def objective(data, config, **params):


    print("Start date:", str(data.index[0].date()))
    # Split a timeseries for forecasting tasks.
    X = np.array(data)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # get start of test data by keeping 2 years of data for training
    test_start_index = data.index.get_indexer([data.index[0] + pd.offsets.DateOffset(years=data_param['train_set'])], method = 'bfill')[0]
    test_size = int(len(data) - test_start_index)

    input_data = {}

    # split data into train and test
    for i in range(data_param["H"]):
        input_data[i+1] = list(to_forecasting(X, forecast= i+1, test_size = test_size))
        input_data[i+1][2] = input_data[i+1][2][:,:-2]
        input_data[i+1][3] = input_data[i+1][3][:,:-2]


    scaler = MinMaxScaler()
    X = np.array(data)
    X = scaler.fit_transform(X[:,:-2])

    for i in range(data_param["H"]):
        input_data[i+1][3] = scaler.inverse_transform(input_data[i+1][3])


    # Split timeseries for forecasting returns squared
    if data_param['diagonal_cov']:
        X_2 = np.array(data**2)
        scaler = MinMaxScaler()
        X_2 = scaler.fit_transform(X_2)


        input_data_2 = {}

        # split data into train and test
        for i in range(data_param["H"]):
            input_data_2[i+1] = list(to_forecasting(X_2, forecast= i+1, test_size = test_size))
            input_data_2[i+1][2] = input_data_2[i+1][2][:,:-2]
            input_data_2[i+1][3] = input_data_2[i+1][3][:,:-2]



        scaler = MinMaxScaler()
        X_2 = np.array(data**2)
        X_2 = scaler.fit_transform(X_2[:,:-2])

        for i in range(data_param["H"]):
            input_data_2[i+1][3] = scaler.inverse_transform(input_data_2[i+1][3])


    test_index = get_test_timeindex(data.index, input_data[i+1][3])
    instances = config["instances_per_trial"]

    variable_seed = params["seed"]

    losses = []; information_ratio = []; profit = []; instances_performed = 0
    try:

        for n in range(instances):
            # Build your model given the input parameters
            reservoir = Reservoir(params["units"],
                                sr=params["spectral_radius"],
                                lr=params["leak_rate"],
                                input_scaling = params["input_scaling"],
                                seed=variable_seed)

            readout = Ridge(ridge=params["ridge"])

            predictions = {}
            # Train your model and test your model.
            for i in range(data_param["H"]):
                model = reservoir >> readout
                predictions[i+1] = model.fit(input_data[i+1][0], input_data[i+1][2]) \
                            .run(input_data[i+1][1])
                predictions[i+1] = scaler.inverse_transform(predictions[i+1])

                print("RMSE:", rmse(input_data[i+1][3], predictions[i+1]), "R^2 score:", rsquare(input_data[i+1][3], predictions[i+1]))
                predictions[i+1] = pd.DataFrame(data = predictions[i+1], index = test_index, columns = list(data.columns[:-2]))


            print(predictions[1])
            if data_param['diagonal_cov']:

                predictions_2 = {}
                # Train your model and test your model.
                for i in range(data_param["H"]):
                    model = reservoir >> readout
                    predictions_2[i+1] = model.fit(input_data_2[i+1][0], input_data_2[i+1][2]) \
                                .run(input_data_2[i+1][1])
                    predictions_2[i+1][predictions_2[i+1] < 0] = 0
                    predictions_2[i+1] = scaler.inverse_transform(predictions_2[i+1])

                    print("RMSE:", rmse(input_data_2[i+1][3], predictions_2[i+1]), "R^2 score:", rsquare(input_data_2[i+1][3], predictions_2[i+1]))
                    predictions_2[i+1] = pd.DataFrame(data = predictions_2[i+1], index = test_index, columns = list(data.columns[:-2]))


                print(predictions_2[1])

            # define constraints and set objective
            objective = []
            if data_param['long']:
                constraints = [cvx.LongOnly(), cvx.LeverageLimit(params['leverage']), cvx.TurnoverLimit(params['turnover']), cvx.MaxWeights(params['weight'])]
                for i in range(data_param["H"]):
                    if data_param['diagonal_cov']:
                        objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - params["gamma_risk_" +str(i+1)] * (
                        cvx.DiagonalCovariance(predictions_2[i+1]) + params["kappa_" + str(i+1)] * cvx.RiskForecastError()
                    ) - params["gamma_trade_" +str(i+1)] * cvx.StocksTransactionCost())
                    else:
                        objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - params["gamma_risk_" +str(i+1)] * (
                        cvx.FullCovariance() + params["kappa_" + str(i+1)] * cvx.RiskForecastError()
                    ) - params["gamma_trade_" +str(i+1)] * cvx.StocksTransactionCost())


            else:
                constraints = [cvx.LeverageLimit(params['leverage']), cvx.TurnoverLimit(params['turnover']), cvx.MaxWeights(params['weight']), cvx.MinWeights(-params['weight'])] # longshort
                for i in range(data_param["H"]):
                    if data_param['diagonal_cov']:
                        objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - params["gamma_risk_" +str(i+1)] * (
                        cvx.DiagonalCovariance(predictions_2[i+1]) + params["kappa_" + str(i+1)] * cvx.RiskForecastError()
                    ) - params["gamma_trade_" +str(i+1)] * cvx.StocksTransactionCost() - params["gamma_hold_" +str(i+1)] * cvx.StocksHoldingCost())
                    else:
                        objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - params["gamma_risk_" +str(i+1)] * (
                        cvx.FullCovariance() + params["kappa_" + str(i+1)] * cvx.RiskForecastError()
                    ) - params["gamma_trade_" +str(i+1)] * cvx.StocksTransactionCost() - params["gamma_hold_" +str(i+1)] * cvx.StocksHoldingCost())

            if data_param["soft_constraints"]:
                for i in range(data_param["H"]):
                    for constraint in constraints:
                        objective[i] -= (100 * cvx.SoftConstraint(constraint))
                constraints = []

            # - 0.1 * cvx.ReturnsForecastError(cvx.forecast.HistoricalStandardDeviation)

            # , benchmark = cvx.Uniform()

            policy = cvx.MultiPeriodOptimization(objective, [constraints] * data_param["H"], ignore_dpp = True)

            test_dates = predictions[1].index
            print(str(test_dates[0].date()))

            result = simulator.backtest(policy, start_time = str(test_dates[0].date()), end_time = data_param['date_to'])
            loss = - result.sharpe_ratio
            losses.append(loss)
            information_ratio.append(result.information_ratio)
            profit.append(result.profit)
            instances_performed += 1

            # Change the seed between instances
            variable_seed += 1

    except PortfolioOptimizationError:
        losses = []
        pass
    # Return a dictionary of metrics. The 'loss' key is mandatory when
    # using hyperopt.
    return {'loss': np.mean(losses),
            'information ratio': np.mean(information_ratio),
            'profit': np.mean(profit),
            "RMSE": rmse(input_data[1][3], predictions[1]),
            "R^2 score": rsquare(input_data[1][3], predictions[1]),
            "instances": instances_performed}

def get_best_params(result_path):
    report_path = path.join(result_path, "results")
    results = []
    for file in os.listdir(report_path):
        if path.isfile(path.join(report_path, file)):
            with open(path.join(report_path, file), "r") as f:
                results.append(json.load(f))
    id = np.nanargmin([r["returned_dict"]['loss'] for r in results])
    return results[id]['current_params']





# from reservoirpy
def plot_results(y_pred, y_test):

    fig = plt.figure(figsize=(15, 7))
    plt.subplot(211)
    plt.plot(np.arange(len(y_pred)), y_pred, lw=3, label="ESN prediction")
    plt.plot(np.arange(len(y_pred)), y_test, linestyle="--", lw=2, label="True value")
    plt.plot(np.abs(y_test - y_pred), label="Absolute deviation")

    plt.legend()
    plt.show(block = False)

def get_predictions(data, data_param, hyperopt_config, instances = 10, hyper_search = True, online = True, seed = 123):
    """Get predictions."""
    # get data 
    # data = get_data(data_param['stocks'])


    if hyper_search:
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, "examples/reservoir_computing/hyper_param_search")
        best = research(objective, data, f"examples/reservoir_computing/hyper_param_search/{hyperopt_config['exp']}.config.json", final_directory)

    param = get_best_params(f"examples/reservoir_computing/hyper_param_search/{hyperopt_config['exp']}")
    print(param)

    print("Start date:", str(data.index[0].date()))
    test_start_index = data.index.get_indexer([data.index[0] + pd.offsets.DateOffset(years=data_param['train_set'])], method = 'bfill')[0]
    test_size = int(len(data) - test_start_index)
    # Split a timeseries for forecasting tasks.
    X = np.array(data)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    input_data = {}

    for i in range(data_param["H"]):
        input_data[i+1] = list(to_forecasting(X, forecast= i+1, test_size = test_size))
        input_data[i+1][2] = input_data[i+1][2][:,:-2]
        input_data[i+1][3] = input_data[i+1][3][:,:-2]


    scaler = MinMaxScaler()
    X = np.array(data)
    X = scaler.fit_transform(X[:,:-2])

    for i in range(data_param["H"]):
        input_data[i+1][3] = scaler.inverse_transform(input_data[i+1][3]) # y_test

    if data_param['diagonal_cov']:

        # Split timeseries for forecasting returns squared

        X_2 = np.array(data**2)
        scaler = MinMaxScaler()
        X_2 = scaler.fit_transform(X_2)


        input_data_2 = {}

        # split data into train and test
        for i in range(data_param["H"]):
            input_data_2[i+1] = list(to_forecasting(X_2, forecast= i+1, test_size = test_size))
            input_data_2[i+1][2] = input_data_2[i+1][2][:,:-2]
            input_data_2[i+1][3] = input_data_2[i+1][3][:,:-2]



        scaler = MinMaxScaler()
        X_2 = np.array(data**2)
        X_2 = scaler.fit_transform(X_2[:,:-2])

    test_index = get_test_timeindex(data.index, input_data[1][3])

    predictions = {}
    predictions_2 = {}
    
    for i in range(data_param["H"]):
        variable_seed = seed
        pred = []
        pred_2 = []
        for j in range(instances):
            reservoir = Reservoir(units = param['units'], input_scaling=param['input_scaling'], sr=param['spectral_radius'],
                      lr=param['leak_rate'], seed=variable_seed)
        
            # if online:
                
            #     readout = FORCE(alpha = param['ridge'])

            #     esn_online = reservoir >> readout

            #     # train and test the model
            #     # esn_online.train(X_train, y_train[:, :-2])
            #     esn_online.train(X_train, y_train)
            #     pred = esn_online.run(X_test)
            # else:
            readout = Ridge(ridge = param['ridge'])
            model = reservoir >> readout


            # Train your model and test your model.
            pred.append(model.fit(input_data[i+1][0], input_data[i+1][2]) \
                           .run(input_data[i+1][1]))
            
            readout = Ridge(ridge = param['ridge'])
            model = reservoir >> readout

            if data_param['diagonal_cov']:
                # Train your model and test your model.
                pred_2.append(model.fit(input_data_2[i+1][0], input_data_2[i+1][2]) \
                            .run(input_data_2[i+1][1]))
                pred_2[j][pred_2[j] < 0 ] = 0 
            # Change the seed between instances
            variable_seed += 1
            
        pred = np.mean(pred, axis = 0)
        predictions[i+1] = scaler.inverse_transform(pred)

        print("RMSE:", rmse(input_data[i+1][3], predictions[i+1]), "R^2 score:", rsquare(input_data[i+1][3], predictions[i+1])) 
        predictions[i+1] = pd.DataFrame(data = predictions[i+1], index = test_index, columns = list(data.columns[:-2])) 

        if data_param['diagonal_cov']:
            pred_2 = np.mean(pred_2, axis = 0)
            predictions_2[i+1] = scaler.inverse_transform(pred_2)
            predictions_2[i+1] = pd.DataFrame(data = predictions_2[i+1], index = test_index, columns = list(data.columns[:-2])) 

        
       # plot sample results
    plot_results(np.array(predictions[1].iloc[:,1]), input_data[1][3][:,1])

    print(predictions[1])

    return predictions, predictions_2

# set up the reservoir and get forecasted returns 1 day ahead
# new_data = get_rescaled_returns(data)
predictions, predictions_2 = get_predictions(data, data_param, hyperopt_config, instances = 3, hyper_search = hyperopt_config['hyper_search'], online = True, seed = 123)

param = get_best_params(f"examples/reservoir_computing/hyper_param_search/{hyperopt_config['exp']}")

# define constraints and set objective
objective = []
if data_param['long']:
    constraints = [cvx.LongOnly(), cvx.LeverageLimit(param['leverage']), cvx.TurnoverLimit(param['turnover']), cvx.MaxWeights(param['weight'])]
    for i in range(data_param["H"]):
        if data_param["diagonal_cov"]:
            objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - param['gamma_risk_' + str(i+1)] * (
            cvx.DiagonalCovariance(predictions_2[i+1]) + param['kappa_' + str(i+1)] * cvx.RiskForecastError()
        ) - param['gamma_trade_' + str(i+1)] * cvx.StocksTransactionCost())
        else:
            objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - param['gamma_risk_' + str(i+1)] * (
            cvx.FullCovariance() + param['kappa_' + str(i+1)] * cvx.RiskForecastError()
        ) - param['gamma_trade_' + str(i+1)] * cvx.StocksTransactionCost())
else:
    constraints = [cvx.LeverageLimit(param['leverage']), cvx.TurnoverLimit(param['turnover']), cvx.MaxWeights(param['weight']), cvx.MinWeights(-param['weight'])] # longshort
    for i in range(data_param["H"]):
        if data_param["diagonal_cov"]:
            objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - param['gamma_risk_' + str(i+1)] * (
            cvx.DiagonalCovariance(predictions_2[i+1]) + param['kappa_' + str(i+1)] * cvx.RiskForecastError()
        ) - param['gamma_trade_' + str(i+1)] * cvx.StocksTransactionCost() - param["gamma_hold_" +str(i+1)] * cvx.StocksHoldingCost())
        else: 
            objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - param['gamma_risk_' + str(i+1)] * (
            cvx.FullCovariance() + param['kappa_' + str(i+1)] * cvx.RiskForecastError()
        ) - param['gamma_trade_' + str(i+1)] * cvx.StocksTransactionCost() - param["gamma_hold_" +str(i+1)] * cvx.StocksHoldingCost())
            
if data_param["soft_constraints"]:
    for i in range(data_param["H"]):
        for constraint in constraints:
            objective[i] -= (100 * cvx.SoftConstraint(constraint))
    constraints = []

      
# - 0.1 * cvx.ReturnsForecastError(cvx.forecast.HistoricalStandardDeviation)

# , benchmark = cvx.Uniform()

policy = cvx.MultiPeriodOptimization(objective, [constraints] * data_param["H"], ignore_dpp = True)
breakpoint()
print(len(predictions[1].columns))
print(predictions[1].columns)
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