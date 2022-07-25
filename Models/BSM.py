import pandas as pd
from py_vollib_vectorized import vectorized_black_scholes_merton as vbsm
from py_vollib_vectorized import greeks
import numpy as np
from DataGetters.VolatilitySurface import IceVolat
from sqlalchemy import create_engine


def option_calc(strike: float, option_type: str, date: str, contract_start_date: str, atm_price:float=None):
    """
    Function for option's premia calculation.
    :param strike:
    :param option_type: 'p' or 'c'
    :param date: date of volatility&forwards in format '%Y-%m-%d'. For Example '2022-01-01'
    :param contract_start_date: delivery start date in format '%Y-%m-%d'. For Example '2022-01-01'
    :return: float
    """

    date_range = int((np.datetime64(contract_start_date) - np.datetime64(date)) / np.timedelta64(1, 'D')) + 15

    beg_date = np.datetime64(contract_start_date[:10] + str(' 06:00:00'))

    volatility_interpol = IceVolat(str(date)[:10], 'all', 'all')

    volatility_interpol = volatility_interpol.interpolation()
    engine = create_engine('postgresql://postgres:!QAZ2wsx@10.18.22.217/analytics_base', echo=False)

    forward_curve = pd.read_sql(
        f"SELECT * FROM f_get_forward_curve('{'TTF'}','{date}')",
        con=engine)

    if atm_price is None:
        under_price = forward_curve[forward_curve['beg_date'] == beg_date]['value'].to_numpy()[0]
    else:
        under_price = atm_price

    q = 0
    r = 0

    sigma = volatility_interpol[contract_start_date[:10]](strike)

    premium = vbsm(option_type, under_price, strike, (date_range) / 365, r, sigma, q, return_as='numpy.array')
    delta = greeks.delta(option_type,under_price, strike,(date_range) / 365, r,sigma, q=q, return_as='numpy.array')
    gamma = greeks.gamma(option_type, under_price, strike, (date_range) / 365, r, sigma, q=q, return_as='numpy.array')
    theta = greeks.theta(option_type, under_price, strike, (date_range) / 365, r, sigma, q=q, return_as='numpy.array')
    vega = greeks.vega(option_type, under_price, strike, (date_range) / 365, r, sigma, q=q, return_as='numpy.array')

    return {'atm':under_price,'premium':premium[0],'delta':delta[0],'gamma':gamma[0],'theta':theta[0],'vega':vega[0]}

if __name__=='__main__':
    option_calc(strike=86.835, option_type = 'p', date='2022-05-23', contract_start_date = '2022-08-01')
    option_calc(strike=86.835, option_type = 'c', date='2022-05-23', contract_start_date = '2022-08-01')
