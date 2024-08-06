"""Example of hyper-parameter search.

This example implements random hyper-parameter search.
Both MPO hyper-parameters and ESN hyper-parameters are tuned to get the best Sharpe ratio.
Hyper-parameter search spaces are specified in config.py file, together with other settings. 

To run this, you need to install ``reservoirpy`` and ``hyperopt``.
"""
# In the root directory of the development environment: python -m examples.reservoir_computing.reservoirpy_MPO

import os
import json
import sys


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .functions import get_best_params, get_predictions, simulator, data_full
from .config import data_param, hyperopt_config, risk_model, reservoir_param

import cvxportfolio as cvx

def main() -> int:

    # save the configuration in a JSON file
    # each file will begin with a number corresponding to the current experimentation run number.
    with open(f"examples/reservoir_computing/hyper_param_search/{hyperopt_config['exp']}.config.json", "w+") as f:
        json.dump(hyperopt_config, f)


    # set up the reservoir and get forecasted returns
    predictions, predictions_2 = get_predictions(data_full, data_param, reservoir_param, hyperopt_config, instances = 5, hyper_search = data_param['hyper_search'], fixed_param = False, online = data_param['online'], seed = 123)

    param = get_best_params(f"examples/reservoir_computing/hyper_param_search/{hyperopt_config['exp']}")

    # define constraints and set objective
    objective = []
    if data_param['long']:
        constraints = [cvx.LongOnly(), cvx.LeverageLimit(param['leverage']), cvx.TurnoverLimit(param['turnover']), cvx.MaxWeights(param['weight'])]
        for i in range(data_param["H"]):
            if data_param["diagonal_cov"]:
            #     objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - param['gamma_risk_' + str(i+1)] * (
            #     cvx.DiagonalCovariance(predictions_2[i+1]) + param['kappa_' + str(i+1)] * cvx.RiskForecastError()
            # ) - param['gamma_trade_' + str(i+1)] * cvx.StocksTransactionCost())
            # else:
            #     objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - param['gamma_risk_' + str(i+1)] * (
            #     risk_model + param['kappa_' + str(i+1)] * cvx.RiskForecastError()
            # ) - param['gamma_trade_' + str(i+1)] * cvx.StocksTransactionCost())

                objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - param['gamma_risk'] * (
                cvx.DiagonalCovariance(predictions_2[i+1]) + param['kappa'] * cvx.RiskForecastError()
            ) - param['gamma_trade'] * cvx.StocksTransactionCost())
            else:
                objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - param['gamma_risk'] * (
                risk_model + param['kappa'] * cvx.RiskForecastError()
            ) - param['gamma_trade'] * cvx.StocksTransactionCost())     
    else:
        constraints = [cvx.LeverageLimit(param['leverage']), cvx.TurnoverLimit(param['turnover']), cvx.MaxWeights(param['weight']), cvx.MinWeights(-param['weight'])] # longshort
        for i in range(data_param["H"]):
            if data_param["diagonal_cov"]:
            #     objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - param['gamma_risk_' + str(i+1)] * (
            #     cvx.DiagonalCovariance(predictions_2[i+1]) + param['kappa_' + str(i+1)] * cvx.RiskForecastError()
            # ) - param['gamma_trade_' + str(i+1)] * cvx.StocksTransactionCost() - param["gamma_hold_" +str(i+1)] * cvx.StocksHoldingCost())
            # else: 
            #     objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - param['gamma_risk_' + str(i+1)] * (
            #     risk_model + param['kappa_' + str(i+1)] * cvx.RiskForecastError()
            # ) - param['gamma_trade_' + str(i+1)] * cvx.StocksTransactionCost() - param["gamma_hold_" +str(i+1)] * cvx.StocksHoldingCost())

                objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - param['gamma_risk'] * (
                cvx.DiagonalCovariance(predictions_2[i+1]) + param['kappa'] * cvx.RiskForecastError()
            ) - param['gamma_trade'] * cvx.StocksTransactionCost() - param["gamma_hold"] * cvx.StocksHoldingCost())
            else: 
                objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - param['gamma_risk'] * (
                risk_model + param['kappa'] * cvx.RiskForecastError()
            ) - param['gamma_trade'] * cvx.StocksTransactionCost() - param["gamma_hold"] * cvx.StocksHoldingCost())
                
    if data_param["soft_constraints"]:
        for i in range(data_param["H"]):
            for constraint in constraints:
                objective[i] -= (100 * cvx.SoftConstraint(constraint))
        constraints = []


    policy = cvx.MultiPeriodOptimization(objective, [constraints] * data_param["H"], benchmark = cvx.Uniform(), ignore_dpp = True)
    print(len(predictions[1].columns))
    print(predictions[1].columns)
    test_dates = predictions[1].index
    print(str(test_dates[0].date()))

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
    print("MPO reservoirpy excess risk (annualized):", results[0].annualized_average_excess_return)
    print("MPO reservoirpy excess return (annualized):", results[0].annualized_excess_volatility)

    print("Uniform allocation profit:", results[1].profit)
    print("Uniform allocation sharpe ratio:", results[1].sharpe_ratio)
    print("Uniform allocation information ratio:", results[1].information_ratio)
    print("Uniform allocation excess risk (annualized):", results[1].annualized_average_excess_return)
    print("Uniform allocation excess return (annualized):", results[1].annualized_excess_volatility)

    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main())
