import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import time

from reservoirpy.observables import rmse

from .functions import get_predictions, data, data_full
from .config import data_param, hyperopt_config, reservoir_param

# python -m examples.reservoir_computing.reservoirpy_forecasts


def main() -> int:

    start = time.time()

    reservoir_param_new = reservoir_param
    # units = [5, 10, 15, 25, 35, 60, 70]
    # spectral_radius = [0.01, 0.1, 0.5, 0.9, 1, 1.5, 5]
    leak_rate = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
    # input_scaling = [0.1, 0.5, 0.7, 0.9]

    test_start_index = data.index.get_indexer([data.index[0] + pd.offsets.DateOffset(years=data_param['train_set'])], method = 'bfill')[0]
    test_size = int(len(data) - test_start_index)

    input_data = {}
    X = np.array(data)

    for i in range(data_param["H"]):
        input_data[i+1] = [X[:test_start_index-1], X[test_start_index-1:-1], X[i+1:test_start_index-1+i+1], []] # X[:test_start_index]
        input_data[i+1][2] = input_data[i+1][2][:,:-2] # y_train
        input_data[i+1][3] = np.array(data_full)[test_start_index-1+i+1:test_start_index-1+i+1+test_size,:-2]
    input_data_2 = {}
    X_2 = np.array(data**2)

    # split data into train and test
    for i in range(data_param["H"]):
            input_data_2[i+1] = [X_2[:test_start_index-1], X_2[test_start_index-1:-1], X_2[i+1:test_start_index-1+i+1], []] # X_2[:test_start_index]
            input_data_2[i+1][2] = input_data_2[i+1][2][:,:-2]
            input_data_2[i+1][3] = np.array(data_full**2)[test_start_index-1+i+1:test_start_index-1+i+1+test_size,:-2]

    results = {}
    results_2 = {}

    for u in leak_rate:

        reservoir_param_new['leak_rate'] = u
        print(reservoir_param_new)

        predictions, predictions_2 = get_predictions(data_full, data_param, reservoir_param_new, hyperopt_config, instances = 5, hyper_search = False, fixed_param = True, online = data_param['online'], seed = 123)

        for i in range(data_param["H"]):
            predictions[i+1] = np.array(predictions[i+1]) 
            predictions_2[i+1] = np.array(predictions_2[i+1]) 
            results[f"RMSE_H_{i+1}_units_{u}"] = rmse(input_data[i+1][3], predictions[i+1])
            results_2[f"RMSE_H_{i+1}_units_{u}"] = rmse(input_data_2[i+1][3], predictions_2[i+1])
            print(f"RMSE_H_{i+1}:", rmse(input_data[i+1][3], predictions[i+1]))
            print(f"RMSE_H_{i+1}:", rmse(input_data_2[i+1][3], predictions_2[i+1]))  
        

    print(results)
    print(results_2)
    print([round(x, 4) for x in results.values()])
    print([round(x, 4) for x in results_2.values()])
    end = time.time()
    print(end-start)
    plt.show()

if __name__ == '__main__':
    sys.exit(main())