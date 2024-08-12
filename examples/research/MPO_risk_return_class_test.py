"""Example of back-tests with predictions obtained using reservoirpy.

This example uses PredictedReturns from test_class_forecast.py.
This example tests different choices of gamma_risk parameter for different planning horizons.
Depending on the config.py settings we can test different strategies. 

    To run this, you need to install ``reservoirpy``.
"""

import json
import sys
import time


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .functions import simulator
from .config import data_param, reservoir_param
from .test_class_forecasts import PredictedReturns

import cvxportfolio as cvx

# python -m examples.reservoir_computing.MPO_risk_return_test

param = {'turnover': 0.1, 'weight': 0.2, 'gamma_trade': 5, 'gamma_hold': 5, 'kappa': 0}
if data_param['long']:
    param['leverage'] = 1
else:
    param['leverage'] = 1
gamma_risks = [1, 5, 10, 25, 50, 100, 200]
units = 60
returns_error = 0 * cvx.ReturnsForecastError()
risk_model = cvx.FactorModelCovariance(num_factors=10)
return_forecaster = PredictedReturns(units)
H = [1, 2, 5, 10]

def main() -> int:

    
    start = time.time()

    results_df = {}

    for h in H:
        data_param["H"] = h
        print(data_param)
        policies = []
        for gamma_risk in gamma_risks:
            # define constraints and set objective
            if data_param['long']:
                constraints = [cvx.LongOnly(), cvx.LeverageLimit(param['leverage']), cvx.TurnoverLimit(param['turnover']), cvx.MaxWeights(param['weight'])]
                if data_param["diagonal_cov"]:
                    if data_param['default_pred2']:
                        if data_param['default_pred']:
                            objective  = cvx.ReturnsForecast() - returns_error - gamma_risk * (
                            cvx.DiagonalCovariance() + param['kappa'] * cvx.RiskForecastError()
                            ) - param['gamma_trade'] * cvx.StocksTransactionCost()
                        else:
                            objective = cvx.ReturnsForecast(r_hat = return_forecaster) - returns_error - gamma_risk * (
                            cvx.DiagonalCovariance() + param['kappa'] * cvx.RiskForecastError()
                            ) - param['gamma_trade'] * cvx.StocksTransactionCost()
                    else:
                        if data_param['default_pred']:
                            objective = cvx.ReturnsForecast() - returns_error - gamma_risk * (
                            cvx.DiagonalCovariance() + param['kappa'] * cvx.RiskForecastError()
                            ) - param['gamma_trade'] * cvx.StocksTransactionCost()
                        else:
                            objective = cvx.ReturnsForecast(r_hat = return_forecaster) - returns_error - gamma_risk * (
                            cvx.DiagonalCovariance() + param['kappa'] * cvx.RiskForecastError()
                            ) - param['gamma_trade'] * cvx.StocksTransactionCost()
                else:
                    if data_param['default_pred']:
                        objective = cvx.ReturnsForecast() - returns_error - gamma_risk * (
                        risk_model + param['kappa'] * cvx.RiskForecastError()
                        ) - param['gamma_trade'] * cvx.StocksTransactionCost()
                    else:
                        objective = cvx.ReturnsForecast(r_hat = return_forecaster) - returns_error - gamma_risk * (
                        risk_model + param['kappa'] * cvx.RiskForecastError()
                        ) - param['gamma_trade'] * cvx.StocksTransactionCost()
            else:
                constraints = [cvx.LeverageLimit(param['leverage']), cvx.TurnoverLimit(param['turnover']), cvx.MaxWeights(param['weight']), cvx.MinWeights(-param['weight'])] # longshort
                for i in range(data_param["H"]):
                    if data_param["diagonal_cov"]:
                        if data_param['default_pred2']:
                            if data_param['default_pred']:
                                objective = cvx.ReturnsForecast() - returns_error - gamma_risk * (
                                cvx.DiagonalCovariance() + param['kappa'] * cvx.RiskForecastError()
                                ) - param['gamma_trade'] * cvx.StocksTransactionCost() - param["gamma_hold"] * cvx.StocksHoldingCost()
                            else:
                                objective = cvx.ReturnsForecast(r_hat = return_forecaster) - returns_error - gamma_risk * (
                                cvx.DiagonalCovariance() + param['kappa'] * cvx.RiskForecastError()
                                ) - param['gamma_trade'] * cvx.StocksTransactionCost() - param["gamma_hold"] * cvx.StocksHoldingCost()
                        else:
                            if data_param['default_pred']:
                                objective = cvx.ReturnsForecast() - returns_error - gamma_risk * (
                                cvx.DiagonalCovariance() + param['kappa'] * cvx.RiskForecastError()
                                ) - param['gamma_trade'] * cvx.StocksTransactionCost() - param["gamma_hold"] * cvx.StocksHoldingCost()
                            else:
                                objective = cvx.ReturnsForecast(r_hat = return_forecaster) - returns_error - gamma_risk * (
                                cvx.DiagonalCovariance() + param['kappa'] * cvx.RiskForecastError()
                                ) - param['gamma_trade'] * cvx.StocksTransactionCost() - param["gamma_hold"] * cvx.StocksHoldingCost()
                    else: 
                        if data_param['default_pred']:
                            objective = cvx.ReturnsForecast() - returns_error - gamma_risk * (
                            risk_model + param['kappa'] * cvx.RiskForecastError()
                            ) - param['gamma_trade'] * cvx.StocksTransactionCost() - param["gamma_hold"] * cvx.StocksHoldingCost()
                        else:
                            objective = cvx.ReturnsForecast(r_hat = return_forecaster) - returns_error - gamma_risk * (
                            risk_model + param['kappa'] * cvx.RiskForecastError()
                            ) - param['gamma_trade'] * cvx.StocksTransactionCost() - param["gamma_hold"] * cvx.StocksHoldingCost()
                        
            if data_param["soft_constraints"]:
                for i in range(data_param["H"]):
                    for constraint in constraints:
                        objective -= (100 * cvx.SoftConstraint(constraint))
                constraints = []

                
            policies.append(cvx.MultiPeriodOptimization(objective, constraints, planning_horizon = data_param["H"], benchmark = cvx.Uniform(), ignore_dpp = True))

        results = simulator.backtest_many(policies, start_time = '2017-01-03', end_time = data_param['date_to'])
        results_df[h] = results
        print(time.time()-start)

    print(time.time()-start)

    result_uniform = simulator.backtest(cvx.Uniform(), start_time ='2017-01-03', end_time = data_param['date_to'])

    MPO_results = {}
    MPO_results['reservoir_param'] = reservoir_param
    for h in H: 
        print(f"MPO H = {h} reservoirpy profits:", [result.profit for result in results_df[h]])
        print(f"MPO H = {h} reservoirpy sharpe ratios:", [result.sharpe_ratio for result in results_df[h]])
        MPO_result = {'sharpe_ratios':  [result.sharpe_ratio for result in results_df[h]], 'profits': [result.profit for result in results_df[h]],
                      'risks': [result.annualized_excess_volatility for result in results_df[h]], 'returns': [result.annualized_average_excess_return for result in results_df[h]]}
        MPO_results[f'MPO_H_{h}'] = MPO_result

    name = f"long_{str(data_param['long'])[0]}_diagonal_{str(data_param['diagonal_cov'])[0]}_defpred_{str(data_param['default_pred'])[0]}_defpred2_{str(data_param['default_pred2'])[0]}_soft_{str(data_param['soft_constraints'])[0]}_turnover_{str(param['turnover'])[2:]}_weight_{str(param['weight'])[2:]}_trade_{str(param['gamma_trade'])}_hold_{str(param['gamma_hold'])}"
    # with open(f"examples/reservoir_computing/results_risk_return/lr_0{str(reservoir_param['leak_rate'])[-1]}/{name}.json", "w+") as f:
    #     json.dump(MPO_results, f)

    plt.figure()
    for h in H:


        plt.plot(
            [result.annualized_excess_volatility for result in results_df[h]],
            [result.annualized_average_excess_return for result in results_df[h]],
            '.-', label='H = %g'%h
            )
    plt.scatter([result_uniform.annualized_excess_volatility],
    [result_uniform.annualized_average_excess_return],
    label='Uniform (1/n) allocation'
    )
    plt.legend()
    plt.title(f'Back-Test Result (Out-Of-Sample)')
    plt.xlabel('Excess risk (annualized)')
    plt.ylabel('Excess return (annualized)')

    # plt.savefig(f"examples/reservoir_computing/plots/lr_0{str(reservoir_param['leak_rate'])[-1]}/{name}")

    plt.figure()
    for h in H:


        plt.plot(
            [result.annualized_excess_volatility for result in results_df[h]],
            [result.annualized_average_excess_return for result in results_df[h]],
            '.-', label='H = %g'%h
            )
    plt.legend()
    plt.title(f'Back-Test Result (Out-Of-Sample)')
    plt.xlabel('Excess risk (annualized)')
    plt.ylabel('Excess return (annualized)')

    # plt.savefig(f"examples/reservoir_computing/plots/lr_0{str(reservoir_param['leak_rate'])[-1]}/{name}_no_u")



    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main())