"""Functions needed for hyper-parameters search and getting data and forecasts.

"""

import os
from os import path
import json


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from reservoirpy.nodes import Reservoir, Ridge, FORCE
from reservoirpy.observables import rmse, rsquare
from sklearn.preprocessing import MinMaxScaler
from reservoirpy.hyper import research
from cvxportfolio.errors import ProgramInfeasible
from .config import data_param, risk_model

import cvxportfolio as cvx

# get data
def get_data(stocks_list, keep_stocks = True, date_from = '2019-05-10'):
    # Get a data frame with stocks returns, VIX and interest rate to use to forecast returns
    df = cvx.DownloadedMarketData(stocks_list).returns
    df = df.iloc[:-1,:]
    if keep_stocks:
        df.dropna(inplace = True)
    dates = df.index[(df.index > date_from)]
    df = df.loc[dates]
    if not keep_stocks:
        df.dropna(axis = 1, inplace = True)
        print(len(df.columns))
        print(df.columns)
    df.index = pd.to_datetime(df.index.date)
    df.rename(columns={"USDOLLAR": "rate"}, inplace = True)
    VIX = pd.DataFrame({'VIX':cvx.YahooFinance('^VIX').data.open})
    VIX.index = pd.to_datetime(VIX.index.date)
    df = df.merge(VIX, left_index = True, right_index = True, how = 'left')
    df.index = dates
    print(df)
    return df

data_full = get_data(data_param['stocks'], keep_stocks = data_param['keep_stocks'], date_from = data_param['date_from'])
# define simulator
data = data_full.loc[(data_full.index <= data_param['date_to'])]
simulator = cvx.StockMarketSimulator(list(data.columns[:-2]))

# define objective function, based on: https://reservoirpy.readthedocs.io/en/latest/user_guide/hyper.html

def objective(data, config, **params):

    data_full = data.copy()
    data = data.loc[(data.index <= data_param['date_to'])]

    print("Start date:", str(data.index[0].date()))
    # Split a timeseries for forecasting tasks.
    X = np.array(data)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # get start of test data by keeping 2 years of data for training
    test_start_index = data.index.get_indexer([data.index[0] + pd.offsets.DateOffset(years=config['train_set'])], method = 'bfill')[0]
    test_size = int(len(data) - test_start_index)

    input_data = {}

    # split data into train and test
    for i in range(config["H"]):
        input_data[i+1] = [X[:test_start_index-1], X[test_start_index-1:-1], X[i+1:test_start_index-1+i+1], []] 
        input_data[i+1][2] = input_data[i+1][2][:,:-2]
        input_data[i+1][3] = np.array(data_full)[test_start_index-1+i+1:test_start_index-1+i+1+test_size,:-2]


    scaler = MinMaxScaler()
    X = np.array(data)
    X = scaler.fit_transform(X[:,:-2])

    # Split timeseries for forecasting returns squared
    if config['diagonal_cov']:
        X_2 = np.array(data**2)
        scaler_2 = MinMaxScaler()
        X_2 = scaler_2.fit_transform(X_2)


        input_data_2 = {}

        # split data into train and test
        for i in range(config["H"]):
            input_data_2[i+1] = [X_2[:test_start_index-1], X_2[test_start_index-1:-1], X_2[i+1:test_start_index-1+i+1], []]
            input_data_2[i+1][2] = input_data_2[i+1][2][:,:-2]
            input_data_2[i+1][3] = np.array(data_full**2)[test_start_index-1+i+1:test_start_index-1+i+1+test_size,:-2]



        scaler_2 = MinMaxScaler()
        X_2 = np.array(data**2)
        X_2 = scaler_2.fit_transform(X_2[:,:-2])


    test_index = data.index[test_start_index:]
    instances = config["instances_per_trial"]

    variable_seed = params["seed"]

    policies = []
    
    for n in range(instances):
        # Build your model given the input parameters
        reservoir = Reservoir(params["units"],
                            sr=params["spectral_radius"],
                            lr=params["leak_rate"],
                            input_scaling = params["input_scaling"],
                            seed=variable_seed)

        predictions = {}
        # Train your model and test your model.
        for i in range(config["H"]):

            if config['online']:
            
                readout = FORCE(alpha = params['ridge'])

                esn_online = reservoir >> readout

                # train and test the model
                esn_online.train(input_data[i+1][0], input_data[i+1][2])
                predictions[i+1] = esn_online.run(input_data[i+1][1])
            else:
                readout = Ridge(ridge = params['ridge'])

                model = reservoir >> readout

                predictions[i+1] = model.fit(input_data[i+1][0], input_data[i+1][2]) \
                            .run(input_data[i+1][1])
            predictions[i+1] = scaler.inverse_transform(predictions[i+1])

            print("RMSE:", rmse(input_data[i+1][3], predictions[i+1]), "R^2 score:", rsquare(input_data[i+1][3], predictions[i+1]))
            predictions[i+1] = pd.DataFrame(data = predictions[i+1], index = test_index, columns = list(data.columns[:-2]))


        print(predictions[1])
        if config['diagonal_cov']:

            predictions_2 = {}
            # Train your model and test your model.
            for i in range(config["H"]):
                if config['online']:
                    readout = FORCE(alpha = params['ridge'])

                    esn_online = reservoir >> readout

                    # train and test the model
                    esn_online.train(input_data_2[i+1][0], input_data_2[i+1][2])
                    predictions_2[i+1] = esn_online.run(input_data_2[i+1][1])

                else:
                    readout = Ridge(ridge = params['ridge'])

                    model = reservoir >> readout

                    predictions_2[i+1] = model.fit(input_data_2[i+1][0], input_data_2[i+1][2]) \
                                .run(input_data_2[i+1][1])
                
                predictions_2[i+1][predictions_2[i+1] < 0] = 0

                predictions_2[i+1] = scaler_2.inverse_transform(predictions_2[i+1])

                print("RMSE:", rmse(input_data_2[i+1][3], predictions_2[i+1]), "R^2 score:", rsquare(input_data_2[i+1][3], predictions_2[i+1]))
                predictions_2[i+1] = pd.DataFrame(data = predictions_2[i+1], index = test_index, columns = list(data.columns[:-2]))


            print(predictions_2[1])

        # define constraints and set objective
        objective = []
        if config['long']:
            constraints = [cvx.LongOnly(), cvx.LeverageLimit(params['leverage']), cvx.TurnoverLimit(params['turnover']), cvx.MaxWeights(params['weight'])]
            for i in range(config["H"]):
                if config['diagonal_cov']:
                #     objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - params["gamma_risk_" +str(i+1)] * (
                #     cvx.DiagonalCovariance(predictions_2[i+1]) + params["kappa_" + str(i+1)] * cvx.RiskForecastError()
                # ) - params["gamma_trade_" +str(i+1)] * cvx.StocksTransactionCost())
                # else:
                #     objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - params["gamma_risk_" +str(i+1)] * (
                #     risk_model + params["kappa_" + str(i+1)] * cvx.RiskForecastError()
                # ) - params["gamma_trade_" +str(i+1)] * cvx.StocksTransactionCost())
                    objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - params["gamma_risk"] * (
                    cvx.DiagonalCovariance(predictions_2[i+1]) + params["kappa"] * cvx.RiskForecastError()
                    ) - params["gamma_trade"] * cvx.StocksTransactionCost())
                else:
                    objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - params["gamma_risk"] * (
                    risk_model + params["kappa"] * cvx.RiskForecastError()
                    ) - params["gamma_trade"] * cvx.StocksTransactionCost())


        else:
            constraints = [cvx.LeverageLimit(params['leverage']), cvx.TurnoverLimit(params['turnover']), cvx.MaxWeights(params['weight']), cvx.MinWeights(-params['weight'])] # longshort
            for i in range(config["H"]):
                if config['diagonal_cov']:
                #     objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - params["gamma_risk_" +str(i+1)] * (
                #     cvx.DiagonalCovariance(predictions_2[i+1]) + params["kappa_" + str(i+1)] * cvx.RiskForecastError()
                # ) - params["gamma_trade_" +str(i+1)] * cvx.StocksTransactionCost() - params["gamma_hold_" +str(i+1)] * cvx.StocksHoldingCost())
                # else:
                #     objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - params["gamma_risk_" +str(i+1)] * (
                #     risk_model + params["kappa_" + str(i+1)] * cvx.RiskForecastError()
                # ) - params["gamma_trade_" +str(i+1)] * cvx.StocksTransactionCost() - params["gamma_hold_" +str(i+1)] * cvx.StocksHoldingCost())
                    
                    objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - params["gamma_risk"] * (
                    cvx.DiagonalCovariance(predictions_2[i+1]) + params["kappa"] * cvx.RiskForecastError()
                ) - params["gamma_trade"] * cvx.StocksTransactionCost() - params["gamma_hold"] * cvx.StocksHoldingCost())
                else:
                    objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - params["gamma_risk"] * (
                    risk_model + params["kappa"] * cvx.RiskForecastError()
                ) - params["gamma_trade"] * cvx.StocksTransactionCost() - params["gamma_hold"] * cvx.StocksHoldingCost())

        if config["soft_constraints"]:
            for i in range(config["H"]):
                for constraint in constraints:
                    objective[i] -= (100 * cvx.SoftConstraint(constraint))
            constraints = []

        policies.append(cvx.MultiPeriodOptimization(objective, [constraints] * config["H"], benchmark = cvx.Uniform(), ignore_dpp = True))

        # Change the seed between instances
        variable_seed += 1

    try:
                
        test_dates = predictions[1].index
        print(str(test_dates[0].date()))

        results = simulator.backtest_many(policies, start_time = str(test_dates[0].date()), end_time = config['date_to'])
        losses = [-result.sharpe_ratio for result in results]
        information_ratio = [result.information_ratio for result in results]
        profits = [result.profit for result in results]

    except ProgramInfeasible:
        losses = []; information_ratio = []; profits = []
        pass

    # Return a dictionary of metrics. The 'loss' key is mandatory when
    # using hyperopt.
    return {'loss': np.mean(losses),
            'information ratio': np.mean(information_ratio),
            'profit': np.mean(profits),
            "RMSE": rmse(input_data[1][3], predictions[1]),
            "R^2 score": rsquare(input_data[1][3], predictions[1])}

def get_best_params(result_path):
    report_path = path.join(result_path, "results")
    results = []
    for file in os.listdir(report_path):
        if path.isfile(path.join(report_path, file)):
            with open(path.join(report_path, file), "r") as f:
                results.append(json.load(f))
    id = np.nanargmin([r["returned_dict"]['loss'] for r in results])
    # id = np.nanargmax([r["returned_dict"]['profit'] for r in results])
    return results[id]['current_params']


# from reservoirpy
def plot_results(y_pred, y_test):

    y_pred = pd.DataFrame(data = y_pred, index = y_test.index) 
    # y_pred = y_pred.loc[(y_pred.index > '2019-12-31') & (y_pred.index <= '2022-12-31')]
    # y_test = y_test.loc[(y_test.index > '2019-12-31') & (y_test.index <= '2022-12-31')]

    fig = plt.figure(figsize=(15, 7))
    plt.subplot(211)
    plt.xlabel("$Date$")
    plt.ylabel("$Return$") 
    plt.plot(y_test, linestyle="--", lw=2, label="True value")
    plt.plot(y_pred.iloc[:,0], lw=3, label="ESN prediction")
    # plt.plot(np.abs(y_test - y_pred), label="Absolute deviation")

    plt.legend()
    plt.show(block = False)


def get_predictions(data, data_param, reservoir_param, hyperopt_config, instances = 5, hyper_search = True, fixed_param = False, online = True, seed = 123):
    """Get predictions."""
    # get data 
    data_full = data.copy()
    data = data.loc[(data.index <= data_param['date_to'])]

    if hyper_search:
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, "examples/reservoir_computing/hyper_param_search")
        best = research(objective, data_full, f"examples/reservoir_computing/hyper_param_search/{hyperopt_config['exp']}.config.json", final_directory)

    if fixed_param:
        param = reservoir_param
    else: 
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
        input_data[i+1] = [X[:test_start_index-1], X[test_start_index-1:-1], X[i+1:test_start_index-1+i+1], []]
        input_data[i+1][2] = input_data[i+1][2][:,:-2] # y_train
        input_data[i+1][3] = np.array(data_full)[test_start_index-1+i+1:test_start_index-1+i+1+test_size,:-2]


    scaler = MinMaxScaler()
    X = np.array(data)
    X = scaler.fit_transform(X[:,:-2])

    if data_param['diagonal_cov']:

        # Split timeseries for forecasting returns squared

        X_2 = np.array(data**2)
        scaler_2 = MinMaxScaler()
        X_2 = scaler_2.fit_transform(X_2)


        input_data_2 = {}

        # split data into train and test
        for i in range(data_param["H"]):
            # input_data_2[i+1] = list(to_forecasting(X_2, forecast= i+1, test_size = test_size))
            input_data_2[i+1] = [X_2[:test_start_index-1], X_2[test_start_index-1:-1], X_2[i+1:test_start_index-1+i+1], []] # X_2[:test_start_index]
            input_data_2[i+1][2] = input_data_2[i+1][2][:,:-2]
            input_data_2[i+1][3] = np.array(data_full**2)[test_start_index-1+i+1:test_start_index-1+i+1+test_size,:-2]



        scaler_2 = MinMaxScaler()
        X_2 = np.array(data**2)
        X_2 = scaler_2.fit_transform(X_2[:,:-2])
        
    test_index = data.index[test_start_index:]

    predictions = {}
    predictions_2 = {}
    
    for i in range(data_param["H"]):
        variable_seed = seed
        pred = []
        pred_2 = []
        for j in range(instances):
            reservoir = Reservoir(units = param['units'], input_scaling=param['input_scaling'], sr=param['spectral_radius'],
                      lr=param['leak_rate'], seed=variable_seed)
        
            if online:
                
                readout = FORCE(alpha = param['ridge'])

                esn_online = reservoir >> readout

                # train and test the model
                esn_online.train(input_data[i+1][0], input_data[i+1][2])
                pred.append(esn_online.run(input_data[i+1][1]))
            else:
                readout = Ridge(ridge = param['ridge'])
                model = reservoir >> readout


                # Train your model and test your model.
                pred.append(model.fit(input_data[i+1][0], input_data[i+1][2]) \
                            .run(input_data[i+1][1]))
            
            if data_param['diagonal_cov']:
                # Train your model and test your model.
                if online:
                
                    readout = FORCE(alpha = param['ridge'])

                    esn_online = reservoir >> readout

                    # train and test the model
                    esn_online.train(input_data_2[i+1][0], input_data_2[i+1][2])
                    pred_2.append(esn_online.run(input_data_2[i+1][1]))
                else:
                    readout = Ridge(ridge = param['ridge'])
                    model = reservoir >> readout
                    pred_2.append(model.fit(input_data_2[i+1][0], input_data_2[i+1][2]) \
                                .run(input_data_2[i+1][1]))
                pred_2[j][pred_2[j] < 0 ] = 0 
            # Change the seed between instances
            variable_seed += 1
            
        pred = np.mean(pred, axis = 0)
        predictions[i+1] = scaler.inverse_transform(pred)

        print(f"RMSE_H_{i+1}:", rmse(input_data[i+1][3], predictions[i+1]), "R^2 score:", rsquare(input_data[i+1][3], predictions[i+1])) 
        predictions[i+1] = pd.DataFrame(data = predictions[i+1], index = test_index, columns = list(data.columns[:-2])) 

        if data_param['diagonal_cov']:
            pred_2 = np.mean(pred_2, axis = 0)
            predictions_2[i+1] = scaler_2.inverse_transform(pred_2)
            predictions_2[i+1] = pd.DataFrame(data = predictions_2[i+1], index = test_index, columns = list(data.columns[:-2])) 

        
       # plot sample results
    # for i in range(data_param["H"]):
    for i in range(1):
        dates = data_full.index[test_start_index-1+i+1:test_start_index-1+i+1+test_size]
        plot_results(np.array(predictions[1].iloc[:,0]), data_full.loc[dates].iloc[:,0])

    print(predictions[1])

    return predictions, predictions_2
