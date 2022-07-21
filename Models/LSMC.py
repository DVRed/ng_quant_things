'''
LSMC model, based on Ilya Zimin version.

Functions:
* lsmc - returns contract NPV and execution strategy.
* decision_vector - LSMC helper function. Returns vector that describes exercise strategy.
* shift - LSMC helper function. Shifts all values of an array `lag` steps forward or backward.
* regression_forecast - LSMC helper function. Forecasts future NPV based on current prices.
* infinite_fine - default fine function for violating MAQ constraint.
'''

import numpy as np
from numba import njit

@njit # Fails with parallel=True parameter, but parallel execution shouldn't increase performance here
def decision_vector(simulation_decisions):
    '''LSMC helper function
    Converts decisions matrix (XxZ) of single simulation y into vector of boolean values (Zx1).
    Vector describes whether contract holder chose to receive or reject the gas for each day of single paths.
    X: how many times the buyer decided to receive max_dcq starting from the beginning of the delivery period;
    Y: number of paths;
    Z: number of days in delivery period.'''
    x = 0
    decision_vector = []
    for z in range(simulation_decisions.shape[1]):
        decision_vector.append(simulation_decisions[x, z])
        x = min(x + simulation_decisions[x, z], simulation_decisions.shape[0] - 1)
    return np.array(decision_vector, dtype=np.bool_)


@njit
def shift(array, lag, fill_value=0):
    '''LSMC helper function
    Moves all values in an array `lag` steps on the first axis.'''
    result = np.empty_like(array)
    if lag > 0:
        result[:lag] = fill_value
        result[lag:] = array[:-lag]
    elif lag < 0:
        result[lag:] = fill_value
        result[:lag] = array[-lag:]
    else:
        result[:] = array
    return result


@njit(parallel=True)
def regression_forecast(underlying_prices, strike_prices, npv, z):
    '''LSMC helper function that returns matrix (X,Y) of NPV forecasts for day z from the viewpoint of period z-1.
    Regression model is used to forecast
    NPV'(x,y,z+1) = c + k1*Price(x,y,z) + k2*Price(x,y,z)^2 + k3*DA(x,y,z) + k4*DA(x,y,z)^2.
    Model is fitted simultaneously for all paths Y of the day z.
    k is a matrix (X, 5) of coefficients for each x. @ operator used for matrix multiplication.
    Matrix (5, Y) to the right of @ contains [1, Price, Price^2, DA, DA^2] for each path y.
    Matrices product result in matrix with forecasts of NPV estimates for one day z for all X and Y'''
    # !Changed to Numba-compatible
    regressors = np.ones((5,) + underlying_prices[0, :, z - 1].shape)
    regressors[1] = underlying_prices[0, :, z - 1]
    regressors[2] = underlying_prices[0, :, z - 1] ** 2
    regressors[3] = strike_prices[0, :, z - 1]
    regressors[4] = strike_prices[0, :, z - 1] ** 2
    # regressors = np.array([np.ones(underlying_prices[0, :, z - 1].shape),
    #                        underlying_prices[0, :, z - 1],
    #                        underlying_prices[0, :, z - 1] ** 2,
    #                        strike_prices[0, :, z - 1],
    #                        strike_prices[0, :, z - 1] ** 2]
    #                       )
    finites = np.isfinite(npv[:, 0, z].T)  # Shouldn't depend on simulation number
    # !Changed to Numba-compatible
    #k, r2 = np.linalg.lstsq(regressors.T, npv[finites, :, z].T, rcond=None)[:2]
    k, r2 = np.linalg.lstsq(regressors.T, npv[finites, :, z].T)[:2]
    k = k.T

    forecast = np.zeros((npv.shape[0], npv.shape[1]))
    forecast[finites, :] = k @ regressors
    forecast[np.logical_not(finites), :] = -np.inf
    return forecast


@njit
def infinite_fine(shortfall_days):
    '''Infinite fine function for non-offtaking MAQ, doesnt't matter how many days remained to fulfill MAQ
    (shortfall_days).
    Could be replaced with more suitable formula in lsmc function.'''
    return -np.inf


#!Numba-compatible nan_to_num
@njit(parallel=True)
def nan_to_num(x, nan):
    shape = x.shape
    x = x.ravel()
    x[np.isnan(x)] = nan
    x = x.reshape(shape)
    return x


@njit(parallel=True)
def lsmc(max_acq, min_acq, max_dcq, min_dcq, delivery_days, underlying_paths, strike_paths,
         premium=0.0, fine_function=infinite_fine):
    """
    Each day Z contract holder decides whether to receive or reject gas
    LSMC helps with this decision to maximize contract value when future values of underlying price are unknown.
    Result depends on the fine_function (fine per MWh of non-offtaken MAQ). By default it's -infty, so holder can't
    violate volume constraints. But in fact there's typically no fine in the contract, only prepayment, so fine_function
    is actually conventional.
    Return estimate of contract value for each path ((Y,) array) and exercise strategy of the contract holder for all
    paths Y and days Z (boolean matrix YxZ).
    For simplicity discounting is omitted, interest rate r assumed to be equal to 0.

    Dimensions
    X: how many times the buyer decided to receive max_dcq starting from the beginning of the delivery period
    Y: number of paths
    Z: number of days in delivery period
    """

    assert delivery_days == underlying_paths.shape[1] == strike_paths.shape[1]

    min_receive_days = int(np.ceil(round(max((min_acq - delivery_days * min_dcq) / (max_dcq - min_dcq), 0), 4)))
    max_receive_days = int(np.floor(round((max_acq - delivery_days * min_dcq) / (max_dcq - min_dcq), 4)))
    assert max_receive_days >= min_receive_days

    # !NOTE: для сравнения с LSMC Андрея использовать округления ниже!
    # min_receive_days = round(max((min_acq - delivery_days * min_dcq) / (max_dcq - min_dcq), 0))
    # max_receive_days = round((max_acq - delivery_days * min_dcq) / (max_dcq - min_dcq))

    X = max_receive_days + 1  # +1 to account for x = 0
    Y = underlying_paths.shape[0]
    Z = delivery_days

    # Initialization --------------------------------------------------------------------------------------------------

    # underlying_prices array (XxYxZ) contains X copies of underlying_paths array
    # strike_prices array (XxYxZ) contains X copies of strike_paths array
    # !Changed to Numba-compatible
    #underlying_prices = np.array([underlying_paths] * X)
    underlying_prices = np.zeros(shape=(X, Y, Z))
    for i in range(X):
        underlying_prices[i] = underlying_paths
    #strike_prices = np.array([strike_paths] * X)
    strike_prices = np.zeros(shape=(X, Y, Z))
    for i in range(X):
        strike_prices[i] = strike_paths
        
    # payoffs array (XxYxZ) contains contract cash flows per 1 MWh for each simulation y on day z
    # payoffs are identical across all X
    payoffs = underlying_prices - strike_prices - premium
    # npv array (XxYxZ) contains sum of current and all future payoffs subject to LSMC execution strategy
    # If contract holder receives gas in day z: NPV(x,y,z) = Payoff(x,y,z) * max_dcq + NPV_forecast(x+1,y,z+1)
    # If contract holder rejects gas in day z: NPV(x,y,z) = Payoff(x,y,z) * min_dcq + NPV_forecast(x,y,z+1)
    npv = np.zeros(shape=underlying_prices.shape)
    # npv_forecast array (XxYxZ) contains forecast of NPV for each day z from the viewpoint of day z-1
    # In day z-1 contract holder does not know NPV of the following day, which is
    # required to make the decision. However, he can forecast them.
    # Forecast is based on the following model:
    # NPV'(x,y,z+1) = c + k1*Price(x,y,z) + k2*Price(x,y,z)^2 + k3*DA(x,y,z) + k4*DA(x,y,z)^2
    npv_forecast = np.zeros(shape=underlying_prices.shape)
    # decisions array (XxYxZ) describes whether the customer decided to receive or reject the gas.
    # decisions contains True value for the outcomes when contract holder decides to take dcq_max.
    # !Changed to Numba-compatible
    #decisions = np.zeros(shape=underlying_prices.shape, dtype=bool)
    decisions = np.zeros(shape=(X, Y, Z), dtype=np.bool_)

    # Filling values of the last delivery day z -----------------------------------------------------------------------
    # subtracting one to account for zero-based numbering
    z = Z - 1
    # All x < min_receive_days - 1 do not allow to meet MAQ restrictions.
    # All corresponding values of array are set equal to the fine function
    for i in range(min_receive_days):
        npv[i, :, z] = fine_function(min_receive_days - i) * max_dcq

    decisions[:min_receive_days - 1, :, z] = True
    # x = min_receive_days - 1 allows only receiving gas since minACQ is otherwise violated
    # NPV estimates of those outcomes are equal to the corresponding payoffs * max_dcq
    npv[min_receive_days - 1, :, z] = payoffs[min_receive_days - 1, :, z] * max_dcq
    decisions[min_receive_days - 1, :, z] = True
    # If min_receive_days - 1 < x < max_receive_days - 1, than contract holder can receive as well as reject gas
    # NPV estimates of those outcomes are therefore equal to the corresponding payoffs * max_dcq,
    # but not less than payoffs * min_dcq
    npv[min_receive_days:max_receive_days, :, z] = np.maximum(
        payoffs[min_receive_days:max_receive_days, :, z] * min_dcq,
        payoffs[min_receive_days:max_receive_days, :, z] * max_dcq)
    decisions[min_receive_days:max_receive_days, :, z] = (payoffs[min_receive_days:max_receive_days, :, z] * max_dcq >=
                                                          payoffs[min_receive_days:max_receive_days, :, z] * min_dcq)
    # If x = max_receive_days, contract holder can not choose to receive gas any longer to avoid maxACQ violation
    # NPV estimates of those outcomes are therefore always equal to min_dcq * payoff
    npv[max_receive_days, :, z] = payoffs[max_receive_days, :, z] * min_dcq
    decisions[max_receive_days, :, z] = False

    # Regression model is used to forecast NPV estimate of period z from the viewpoint of period z-1.
    npv_forecast[:, :, z] = regression_forecast(underlying_prices, strike_prices, npv, z)

    # Filling values of all the other days z --------------------------------------------------------------------------
    # Values are filled backwards
    for z in range(Z - 2, -1, -1):
        # Decision of contract holder is based on comparing expected NPV in case of receiving gas
        # (expected_receive_values) and expected NPV in case of rejecting gas (expected_reject_values)
        # expected_receive_values and expected_reject_values are both matrices (XxY) for delivery day z
        # expected_receive_value(x, y, z) = payoff(x, y, z) * max_dcq + NPV forecast (x+1, y, z+1)
        # expected_reject_value(x, y, z) = payoff(x, y, z) * min_dcq + NPV forecast (x, y, z+1)
        # Contract holder can't offtake more than ACQ, so corresponding fine is always infinite
        expected_receive_values = payoffs[:, :, z] * max_dcq + shift(array=npv_forecast[:, :, z + 1], lag=-1,
                                                                     fill_value=-np.inf)
        expected_reject_values = payoffs[:, :, z] * min_dcq + npv_forecast[:, :, z + 1]
        # If expected_receive_values is higher or equal expected_reject_values,
        # contract holder will choose to receive the gas
        decisions[:, :, z] = expected_receive_values >= expected_reject_values

        # Even though decision was made based on expected values,
        # contract holder will receive actual rather than expected payoffs
        receive_values = payoffs[:, :, z] * max_dcq + shift(array=npv[:, :, z + 1], lag=-1, fill_value=-np.inf)
        reject_values = payoffs[:, :, z] * min_dcq + npv[:, :, z + 1]
        # !Changed to Numba-compatible (np.inf * 0 should be 0 here)
        #npv[:, :, z][decisions[:, :, z]] = receive_values[decisions[:, :, z]]
        #npv[:, :, z][np.logical_not(decisions[:, :, z])] = reject_values[np.logical_not(decisions[:, :, z])]
        # !Подумать, как ускорить, чтобы продолжало работать с Numba
        for x in range(X):
            for y in range(Y):
                npv[x, y, z] = receive_values[x, y] if decisions[x, y, z] else reject_values[x, y]


        # Regression model is used to forecast NPV estimate of period z from the viewpoint of period z-1.
        npv_forecast[:, :, z] = regression_forecast(underlying_prices, strike_prices, npv, z)

    # decision_matrix (YxZ) contains decision_vector (1xZ) for each path y.
    # decision_vector contains True values for days when contract holder decided to receive gas
    # !Changed to Numba-compatible
    #decision_matrix = np.array([decision_vector(decisions[:, y, :]) for y in range(Y)])
    decision_matrix = np.zeros(shape=(Y, Z), dtype=np.bool_)
    
    for y in range(Y):
        decision_matrix[y] = decision_vector(decisions[:, y, :])

    if fine_function is infinite_fine:
        pv = (decision_matrix * (max_dcq - min_dcq) * (underlying_paths - strike_paths - premium) +
              min_dcq * (underlying_paths - strike_paths - premium)).sum(axis=1)
        # !Changed to Numba-compatible
        #assert all(np.isclose(npv[0, :, 0], pv))
        assert np.all(np.absolute((npv[0, :, 0] - pv) / pv) < 0.001) or np.all(np.absolute(npv[0, :, 0] - pv) < 0.001)

    return npv[0, :, 0], decision_matrix
