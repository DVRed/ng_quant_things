import pdb
import numpy as np
import pandas as pd
from Models.get_data_from_DB.get_vol_surface import IceVolat
from sqlalchemy import create_engine
import math
from datetime import datetime


class PriceModel:

    def __init__(self, model_type: str, n_simulations: int, delivery_start: np.datetime64, delivery_end: np.datetime64,
                 date_of_pricing: np.datetime64 = np.datetime64('today'),
                 hub: str = 'TTF', date_of_volat: np.datetime64 = np.datetime64('2022-05-16'), seed: datetime = None,
                 mult_for_delta: float = 0, mult_for_volat: float = 0, mult_for_theta: int = 0):
        """
        Это класс для создания ценовых симуляций по заданным инструментам.

        Methods
        ----------
        simple_model()
            Базовая модель симуляций, написана была Зиминым, переведена в класс и в ССТД Девяткиным.

        Attributes
        ----------
        model_type: str
            тип модели для ценовых симуляций. Пример simple_model, risk_model, two_factor_model ....
            на 23.05.2022 реализована simple_model
        n_simulations: int
            Количество ценовых симуляций (>1)
        delivery_start: np.datetime64
            Дата начала поставки по контракту
        delivery_end: np.datetime64
            Дата окончания поставки по контракту
        date_of_pricing: np.datetime64 = np.datetime64('today')
            Дата симуляций
        hub: str = 'TTF'
            Пункт поставки по контракту ('TTF','THE VTP', 'Austria VTP')
        date_of_volat:np.datetime64=np.datetime64('2022-05-16')
            Дата на которую берётся волатильность
        """

        self.model_type = model_type
        self.n_simulations = n_simulations
        self.date_of_pricing = date_of_pricing
        self.sim_date = date_of_pricing
        self.from_date_m = str(self.date_of_pricing - np.timedelta64(300, 'D'))[:10]
        self.to_date_m = str(self.date_of_pricing)[:10]
        self.hub = hub
        self.delivery_start = delivery_start
        self.delivery_end = delivery_end
        self.date_of_volat = date_of_volat
        self.mult_for_delta = mult_for_delta
        self.seed = seed
        self.mult_for_volat = mult_for_volat
        self.mult_for_theta = mult_for_theta

    def simple_model(self):
        """
        Базовая модель симуляций, написана была Зиминым, переведена в класс и в ССТД Девяткиным.
        """

        # месяца для расчетов
        month = list(map(lambda x: int(str(np.datetime64(str(self.delivery_start)[:7]) + np.timedelta64(x, 'M'))[5:7]),
                         range(int((np.datetime64(str(self.delivery_end)[:7]) - np.datetime64(
                             str(self.delivery_start)[:7])) / np.timedelta64(1, 'M')))))

        # Подключение к базе данных
        engine = create_engine('postgresql://postgres:!QAZ2wsx@10.18.22.217/analytics_base', echo=False)

        # Получение данных по фьючам из базы
        futures = pd.read_sql(
            f"SELECT * FROM f_get_all_forwards('{self.from_date_m}','{self.to_date_m}','{self.hub}','{self.delivery_start}')",
            con=engine).pivot(
            index='date',
            columns='beg_date',
            values='value')
        forward_curve = pd.read_sql(
            f"SELECT * FROM f_get_forward_curve('{self.hub}','{self.date_of_volat}')",
            con=engine)
        day_ahead_price = forward_curve[(forward_curve['code'] == 'DA') & (forward_curve['price_type'] == 'midpoint')][
            'value'].to_numpy()[0]

        futures = futures.to_numpy()[:, :len(month)]

        # get lognormal error
        data = np.log(np.float64(np.transpose(futures[-180:]) / np.transpose(futures[-181:-1])))

        # volatilty parse
        ice_class_data = IceVolat(self.date_of_volat, 'all', 'atm')

        month_range = (np.datetime64(self.delivery_start, 'M') -
                       np.datetime64(self.sim_date, 'M')) / np.timedelta64(1, 'M')

        # get vols
        # vols_0 = ice_class_data.get_table().to_numpy()[:, int(date_range) - 1:]
        vols_0 = ice_class_data.get_table().to_numpy()
        vols_0 = vols_0 + vols_0 * self.mult_for_volat
        vols_0 = np.concatenate((vols_0[0], np.repeat(vols_0[0][-1], 20)), axis=0)
        vols_0 = vols_0[int(month_range) - 1:]

        vols_0 = vols_0[:futures.shape[1]]
        '''
        if len(vols_0) < futures.shape[1]:
            vols_0 = vols_0[:futures.shape[1]]
        else:
            vols_0 = vols_0[:futures.shape[1]]
        '''
        # if vols_0.shape
        vols = vols_0 / (365 ** 0.5)

        cov_matrix = np.float64(np.dot(np.diag(vols), np.dot(np.corrcoef(data), np.diag(vols))))

        # range for sims
        dates_range = (self.delivery_end - self.sim_date) / np.timedelta64(1, 'D') - self.mult_for_theta

        if self.seed is None:
            multiv_data = np.random.multivariate_normal(mean=np.float64((vols ** 2) * (-1 / 2)), cov=cov_matrix,
                                                        size=[self.n_simulations, int(dates_range)])
        else:
            rng = np.random.default_rng(int(self.seed.strftime('%Y%m%d')))
            multiv_data = rng.multivariate_normal(mean=np.float64((vols ** 2) * (-1 / 2)), cov=cov_matrix,
                                                  size=[self.n_simulations, int(dates_range)])
        #np.savez('/mnt/teamdocs_ns/TRD_Exchange/sims_2022-07-07_no_seed.npz', multiv_data)
        # cumulative sum
        cumulat = np.cumsum(multiv_data, axis=1)
        # цены по форвардам к использованию
        # print(np.mean(np.float64(futures[-1])))

        prices = (np.float64(futures[-1]) + self.mult_for_delta) * np.exp(cumulat)
        atm_price = np.mean(np.float64(futures[-1]))
        # получаем Day-Ahead котировки
        for_da = prices[:, int((self.delivery_start - self.sim_date - self.mult_for_theta) / np.timedelta64(1, 'D')):int(
            (self.delivery_end - self.sim_date - self.mult_for_theta) / np.timedelta64(1, 'D'))]

        days_of_index = list(map(
            lambda x: np.datetime64(self.delivery_start, 'M') + np.timedelta64(1 * x, 'M') + np.timedelta64(0, 'D'),
            range(math.ceil(len(month) / 1) + 1)))

        days_of_index_repeated = np.cumsum(
            list(map(lambda x: (days_of_index[x + 1] - days_of_index[x]) / np.timedelta64(1, 'D'),
                     range(len(days_of_index) - 1))))

        days_of_index_repeated = np.sort(np.append(days_of_index_repeated, np.array([0]), axis=0))

        spot = list(
            map(lambda x: for_da[:, int(days_of_index_repeated[x]):int(days_of_index_repeated[x + 1]), x],
                range(len(month))))

        spot_prices_sim = []
        for i in range(len(spot[0])):
            temp = []
            for j in range(len(spot)):
                temp.append(spot[j][i])
            spot_prices_sim.append([item for sublist in temp for item in sublist])

        return {'futures': prices, 'spot': np.array(spot_prices_sim), 'da': day_ahead_price, 'atm': atm_price}


if __name__ == '__main__':
    import time

    t_0 = time.time()
    a = PriceModel(model_type='simple',
                   n_simulations=2500,
                   delivery_start=np.datetime64('2022-02-01'),
                   delivery_end=np.datetime64('2024-01-01'),
                   date_of_pricing=np.datetime64('2022-01-08'),
                   hub='TTF',
                   date_of_volat=np.datetime64('2022-01-07')).simple_model()

