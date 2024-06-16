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

risk_model = cvx.FactorModelCovariance(num_factors=10)
# set parameters for getting data
# "date_from": '2019-05-10', '2015-01-01'
data_param = {"stocks": NDX100, 'keep_stocks': False, "date_from": '2015-01-01', "date_to": '2024-05-01',
              "H": 3, "long": False, "diagonal_cov": True, "soft_constraints": False, "train_set": 2, 
              "online": False, "hyper_search": False, 'fixed_param': False} 

# H = planning horizon, keep_stocks: if True we'll keep all stocks listed and use only dates for which all stocks have data, otherwise stocks without data for specified dates will be dropped
# train_set = how many years to take for training, diagonal_cov = whether to use diagonalcovariance, long = long or longshort portfolio, hyper_search: run hyper-parameter optimization or not

# get objectives' parameters 
obj_params = {}
for i in range(data_param["H"]):
    obj_params['kappa_' + str(i+1)] = ["choice", 0, 0.05, 0.1, 0.5]  # covariance forecast error risk parameter
    obj_params['gamma_risk_' + str(i+1)] = ["choice", 0.5, 1, 5, 10, 25, 50]  # risk aversion parameter
    obj_params['gamma_trade_' + str(i+1)] = ["choice", 0.5, 1, 5, 10, 25, 50]  # trading risk aversion factor
    if not data_param['long']:
        obj_params['gamma_hold_' + str(i+1)] = ["choice", 0.5, 1, 5, 10, 25, 50] # holdings aversion parameter

# set parameters for hyperparameter optimization
hyperopt_config = {
    "exp": f"hyperopt_mult_ndx100_keep_{str(data_param['keep_stocks'])[0]}_long_{str(data_param['long'])[0]}_diagonal_{str(data_param['diagonal_cov'])[0]}_{data_param['H']}_soft_{str(data_param['soft_constraints'])[0]}_online_{str(data_param['online'])[0]}", # the experimentation name
    "hp_max_evals": 100,             # the number of differents sets of parameters hyperopt has to try
    "hp_method": "random",           # the method used by hyperopt to choose those sets ("random" or "tpe")
    "seed": 40,                      # the random state seed, to ensure reproducibility
    "instances_per_trial": 3,        # how many random ESN will be tried with each sets of parameters
    **data_param,
    "hp_space": {                    # what are the ranges of parameters explored
        "units": ["choice", 5, 10, 15, 25, 35, 60, 70],             # the number of neurons
        "spectral_radius": ["loguniform", 1e-2, 10],   # the spectral radius is log-uniformly distributed between 1e-2 and 10
        "leak_rate": ["loguniform", 1e-3, 1],  # the leaking rate is log-uniformly distributed between from 1e-3 to 1
        "input_scaling": ["choice", 0.7, 0.9],           # the input scaling is [0.7, 0.9
        "ridge": ["choice", 1e-5, 1e-7, 1e-9],        # the regularization parameter is [1e-5, 1e-7, 1e-9], when using online learning this is alpha (learning rate)
        "seed": ["choice", 123],         # random seed for the ESN initialization
        "weight": ["choice", 0.05, 0.1, 0.2], # weights constraint
        "leverage": ["choice", 1, 3, 5], # leverage limit
        "turnover": ["choice", 0.01, 0.05, 0.1, 0.3, 0.5], # turnover limit
        **obj_params
    }
}

reservoir_param = {
        "units": 35,             # the number of neurons
        "spectral_radius": 0.9, 
        "leak_rate": 0.5, 
        "input_scaling": 0.7,           
        "ridge": 1e-9      
        }
