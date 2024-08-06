import pandas as pd
import numpy as np

from reservoirpy.nodes import Reservoir, Ridge, FORCE

import cvxportfolio as cvx




class PredictedReturns:
    """Expected return using reservoirpy.


    :param units: Number of units for the reservoir.
    :type units: int
    """

    def __init__(self, units=60, sr=0.9, lr=0.5, input_scaling=0.7, seed=123):
        self.units = units
        self.sr = sr
        self.lr = lr
        self.input_scaling = input_scaling
        self.seed = seed

    def values_in_time(self, past_returns, mpo_step, **kwargs):

        reservoir = Reservoir(units=self.units ,
                                    sr=self.sr,
                                    lr=self.lr,
                                    input_scaling = self.input_scaling,
                                    seed=self.seed)


        readout = Ridge(ridge=1e-5)

        model = reservoir >> readout

        past_returns.dropna(inplace = True)

        prediction = model.fit(past_returns.iloc[:-(mpo_step+1), :-1].values, past_returns.iloc[mpo_step+1:, :-1].values) \
                            .run(past_returns.iloc[-1:, :-1].values)

           
        return prediction[0]
