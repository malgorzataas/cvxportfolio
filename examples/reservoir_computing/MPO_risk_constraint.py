"""Example of back-tests with risk term in constraints.

"""
# In the root directory of the development environment: python -m examples.reservoir_computing.MPO_risk_constraint

import sys
import time
import json


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .functions import get_predictions, simulator, data_full
from .config import data_param, hyperopt_config, reservoir_param

import cvxportfolio as cvx

param = {'turnover': 0.1, 'weight': 0.1, 'gamma_trade': 1, 'gamma_hold': 1, 'kappa': 0}
if data_param['long']:
    param['leverage'] = 1
else:
    param['leverage'] = 1
target_daily_variances = [(0.05**2)/252, (0.1**2)/252, (0.2**2)/252]
returns_error = 0 * cvx.ReturnsForecastError()
risk_model = cvx.FactorModelCovariance(num_factors=10)
H = [1, 2, 5, 10]

def main() -> int:


    # set up the reservoir and get forecasted returns 1 day ahead
    predictions, predictions_2 = get_predictions(data_full, data_param, reservoir_param, hyperopt_config, instances = 5, hyper_search = False, fixed_param = True, online = data_param['online'], seed = 123)
    
    start = time.time()

    results_df = {}

    for h in H:
        data_param["H"] = h
        print(data_param)
        policies = []
        for target_daily_variance in target_daily_variances:
            # define constraints and set objective
            objective = []
            if data_param['long']:
                constraints = [cvx.LongOnly(), cvx.LeverageLimit(param['leverage']), cvx.TurnoverLimit(param['turnover']), cvx.MaxWeights(param['weight'])]
                for i in range(data_param["H"]):
                    if data_param["diagonal_cov"]:
                        if data_param['default_pred2']:
                            constraints += [cvx.DiagonalCovariance() <= target_daily_variance]
                            if data_param['default_pred']:
                                objective.append(cvx.ReturnsForecast() - returns_error 
                                                 - param['gamma_trade'] * cvx.StocksTransactionCost())
                            else:
                                objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - returns_error 
                                                 - param['gamma_trade'] * cvx.StocksTransactionCost())
                        else:
                            constraints += [cvx.DiagonalCovariance(predictions_2[i+1]) <= target_daily_variance]
                            if data_param['default_pred']:
                                objective.append(cvx.ReturnsForecast() - returns_error
                                                 - param['gamma_trade'] * cvx.StocksTransactionCost())
                            else:
                                objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - returns_error
                                                 - param['gamma_trade'] * cvx.StocksTransactionCost())
                    else:
                        constraints += [risk_model <= target_daily_variance]
                        if data_param['default_pred']:
                            objective.append(cvx.ReturnsForecast() - returns_error 
                                        - param['gamma_trade'] * cvx.StocksTransactionCost())
                        else:
                            objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - returns_error 
                                        - param['gamma_trade'] * cvx.StocksTransactionCost())
            else:
                constraints = [cvx.LeverageLimit(param['leverage']), cvx.TurnoverLimit(param['turnover']), cvx.MaxWeights(param['weight']), cvx.MinWeights(-param['weight'])] # longshort
                for i in range(data_param["H"]):
                    if data_param["diagonal_cov"]:
                        if data_param['default_pred2']:
                            constraints += [cvx.DiagonalCovariance() <= target_daily_variance]
                            if data_param['default_pred']:
                                objective.append(cvx.ReturnsForecast() - returns_error
                                                 - param['gamma_trade'] * cvx.StocksTransactionCost() - param["gamma_hold"] * cvx.StocksHoldingCost())
                            else:
                                objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - returns_error
                                                 - param['gamma_trade'] * cvx.StocksTransactionCost() - param["gamma_hold"] * cvx.StocksHoldingCost())
                        else:
                            constraints += [cvx.DiagonalCovariance(predictions_2[i+1]) <= target_daily_variance]
                            if data_param['default_pred']:
                                objective.append(cvx.ReturnsForecast() - returns_error
                                                 - param['gamma_trade'] * cvx.StocksTransactionCost() - param["gamma_hold"] * cvx.StocksHoldingCost())
                            else:
                                objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - returns_error
                                                 - param['gamma_trade'] * cvx.StocksTransactionCost() - param["gamma_hold"] * cvx.StocksHoldingCost())
                    else: 
                        constraints += [risk_model <= target_daily_variance]
                        if data_param['default_pred']:
                            objective.append(cvx.ReturnsForecast() - returns_error
                                         - param['gamma_trade'] * cvx.StocksTransactionCost() - param["gamma_hold"] * cvx.StocksHoldingCost())
                        else:
                            objective.append(cvx.ReturnsForecast(r_hat = predictions[i+1]) - returns_error
                                         - param['gamma_trade'] * cvx.StocksTransactionCost() - param["gamma_hold"] * cvx.StocksHoldingCost())
                        
            if data_param["soft_constraints"]:
                for i in range(data_param["H"]):
                    for constraint in constraints:
                        objective[i] -= (100 * cvx.SoftConstraint(constraint))
                constraints = []

                
            # - 0.1 * cvx.ReturnsForecastError(cvx.forecast.HistoricalStandardDeviation)


            policies.append(cvx.MultiPeriodOptimization(objective, [constraints] * data_param["H"], ignore_dpp = True, benchmark = cvx.Uniform()))
        # breakpoint()
        print(len(predictions[1].columns))
        print(predictions[1].columns)
        test_dates = predictions[1].index
        print(str(test_dates[0].date()))

        # result = simulator.backtest(policy, start_time = str(test_dates[0].date()), end_time = data_param['date_to'])

        results = simulator.backtest_many(policies, start_time = str(test_dates[0].date()), end_time = data_param['date_to'])
        results_df[h] = results
        print(time.time()-start)

    print(time.time()-start)

    result_uniform = simulator.backtest(cvx.Uniform(), start_time = str(test_dates[0].date()), end_time = data_param['date_to'])

    # for h in H:

    #     plt.figure()
    #     plt.plot(
    #         [result.annualized_excess_volatility for result in results_df[h]],
    #         [result.annualized_average_excess_return for result in results_df[h]],
    #         '.-',
    #         )
    #     plt.legend()
    #     plt.title(f'Back-Test Result (Out-Of-Sample) H = {h}')
    #     plt.xlabel('Excess risk (annualized)')
    #     plt.ylabel('Excess return (annualized)')


    #     plt.show(block = False)

    
    # plt.figure()

    MPO_results = {}
    MPO_results['reservoir_param'] = reservoir_param
    for h in H: 
        print(f"MPO H = {h} reservoirpy profits:", [result.profit for result in results_df[h]])
        print(f"MPO H = {h} reservoirpy sharpe ratios:", [result.sharpe_ratio for result in results_df[h]])
        MPO_result = {'sharpe_ratios':  [result.sharpe_ratio for result in results_df[h]], 'profits': [result.profit for result in results_df[h]],
                      'risks': [result.annualized_excess_volatility for result in results_df[h]], 'returns': [result.annualized_average_excess_return for result in results_df[h]]}
        MPO_results[f'MPO_H_{h}'] = MPO_result

    name = f"long_{str(data_param['long'])[0]}_diagonal_{str(data_param['diagonal_cov'])[0]}_defpred_{str(data_param['default_pred'])[0]}_defpred2_{str(data_param['default_pred2'])[0]}_soft_{str(data_param['soft_constraints'])[0]}_constraint"
    with open(f"examples/reservoir_computing/results_risk_return/risk_constraint/{name}.json", "w+") as f:
        json.dump(MPO_results, f)


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

    plt.savefig(f"examples/reservoir_computing/plots/risk_constraint/{name}")

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

    plt.savefig(f"examples/reservoir_computing/plots/risk_constraint/{name}_no_u")


    plt.show()

    return 0

if __name__ == '__main__':
    sys.exit(main())