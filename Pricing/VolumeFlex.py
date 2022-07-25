import numpy as np
import pandas as pd
from Models.PriceModel import PriceModel
from Models.IndexCounter import IndexCounter
import time
from Models.LSMC import lsmc
from datetime import datetime


class VolumeFlex:

    def __init__(self, delivery_start: np.datetime64, delivery_end: np.datetime64, delivery_point: str,
                 date_of_pricing: np.datetime64, date_of_volat: np.datetime64, max_acq: float, min_aqc: float,
                 max_dcq: float, min_dcq: float, n_sims: int = 2500, fix_price: float = None,
                 indexes: list = [1, 0, 1]):

        """
        Класс для расчета объёмной гибкости, страйки по которой можно записать в формате (x,y,z) по хабам: TTF, THE VTP, Austria VTP.

        Поставка может начинаться и заканчиваться только в начале и в конце месяца, соответственно.

        Attributes:
        ----------
        delivery_start: np.datetime64
            Дата начала поставки по контракту
        delivery_end: np.datetime64
            Дата окончания поставки по контракту
        delivery_point: str
            Пункт поставки. Возможные варианты - TTF, THE VTP, Austria VTP
        date_of_pricing: np.datetime64
            Дата оценки (влияет на симуляции и подтягивание реальных данных из базы)
        date_of_volat: np.datetime64
            Дата волатильности (влияет на симуляции)
        indexes: list
            Принимает список только целых значений
            Индексы в формате:
            1 - сколько инструментов берем
            2 - лаг
            3 - сколько месяцев усреднения
            4 - сколько месяцев действует индекс (можно скипнуть)
            Пример:indexes = [1,0,1,1] - классический Month-Ahead Index

        Methods:
        -------
        _price_model(n_sims)
            Вызывает ценовую модель
        _index_counter(price_sims)
            Вызывает расчет индекса
        volume_flex_counter(max_acq, min_aqc, max_dcq, min_dcq, n_sims: int = 5000)
            Вызывает 2 предыдущих метода и LSMC. Считает гибкость и подгоняет премию.

        P.S. Потенциально можно и интрументы с кастомным сроком поставки, но я не тестил.
        """

        self._delivery_start = delivery_start
        self._delivery_end = delivery_end
        self._delivery_point = delivery_point
        self._date_of_pricing = date_of_pricing
        self._date_of_volat = date_of_volat
        self._max_acq = max_acq
        self._min_acq = min_aqc
        self._max_dcq = max_dcq
        self._min_dcq = min_dcq
        self._n_sims = n_sims
        self._fix_price = fix_price

        if (3 <= len(indexes) <= 4) != True:
            raise ValueError(indexes)
        else:
            self._indexes = indexes

    def _price_model(self, mult_for_delta: float = 0, date_of_pricing: np.datetime64 = None, mult_for_volat: float = 0,
                     mult_for_theta: int = 0):
        """
        Симулирует цены на газ.

        Attributes:
        ----------
        n_sims: int
            Количество симуляций
            БОЛЬШЕ 7500 НЕ СТАВИТЬ. РАСЧЕТЫ ПАДАЮТ.
        Массив на выходе имеет размерность к-во симуляций Х к-во месячных инструментов Х к-во дней.
        """

        if date_of_pricing is None:
            date_of_pricing = self._date_of_pricing
        else:
            pass

        simulations = PriceModel(model_type='simple',
                                 n_simulations=self._n_sims,
                                 delivery_start=self._delivery_start,
                                 delivery_end=self._delivery_end,
                                 date_of_pricing=date_of_pricing,
                                 hub=self._delivery_point,
                                 date_of_volat=self._date_of_volat,
                                 seed=datetime(2023, 1, 1),
                                 mult_for_delta=mult_for_delta,
                                 mult_for_volat=mult_for_volat,
                                 mult_for_theta=mult_for_theta).simple_model()
        return simulations

    def _index_counter(self, price_sims: np.array, sim_date: np.datetime64 = None,mult_for_theta:int=0):
        """
        Расчитывает индекс/страйки по оцениваемому контракту.

        Attributes:
        ----------
        price_sims: np.array
            Ценовые симуляции (с выходными).
            Массив должен иметь на входе размерность к-во симуляций Х к-во месячных инструментов Х к-во дней
        На выходе получаем размерность размерность к-во симуляций Х к-во дней поставки
        """

        if sim_date is None:
            sim_date = self._date_of_pricing
        else:
            sim_date = sim_date

        strikes = IndexCounter().index_count(contract_start=self._delivery_start,
                                             contract_end=self._delivery_end,
                                             simulation_date=sim_date,
                                             indexes=self._indexes,
                                             data=price_sims,mult_for_theta=mult_for_theta)

        return strikes

    def _lsmc_counter(self, delivery_days: int, strikes: np.array, price_simulations: np.array, fee: float = 0, mult_for_greek:float=None):

        if mult_for_greek is None:
            a = lsmc(max_acq=self._max_acq,
                     min_acq=self._min_acq,
                     max_dcq=self._max_dcq,
                     min_dcq=self._min_dcq,
                     delivery_days=delivery_days,
                     underlying_paths=price_simulations,
                     strike_paths=np.array(strikes),
                     premium=fee)
        else:
            a = lsmc(max_acq=self._max_acq,
                     min_acq=self._max_acq*mult_for_greek,
                     max_dcq=self._max_dcq,
                     min_dcq=self._min_dcq,
                     delivery_days=delivery_days,
                     underlying_paths=price_simulations,
                     strike_paths=np.array(strikes),
                     premium=fee)

        return {'option_cost': np.mean(a[0]), 'premium': a[0]}

    def _derivative_function(self, atm_diff: np.array, premium_fee_0: np.array, x):
        plft = np.polyfit(atm_diff, premium_fee_0, 4)
        return plft[0] * 4 * (x ** 3) + plft[1] * 3 * (x ** 2) + plft[2] * 2 * (x ** 1) + plft[3]

    def count_premium(self, n_sims: int = 2500, custom_sims: np.array = None, custom_index: np.array = None,
                      fix_price: float = None):

        """
        Считает гибкость и подгоняет премию.

        Attributes:
        ----------
        max_acq
            Максимальный суммарный отбор по контракту
        min_aqc
            Минимальный суммарный отбор по контракту
        max_dcq
            Максимальный суточный отбор по контракту
        min_dcq
            Минимальный суточный отбор по контракту
        n_sims: int = 5000, optional
            Количество симуляций
            БОЛЬШЕ 7500 НЕ СТАВИТЬ!!! РАСЧЕТЫ ПАДАЮТ.
        custom_sims: np.array = None

        custom_index: np.array = None

        fix_price:float=None
        """

        delivery_days = int((self._delivery_end - self._delivery_start) / np.timedelta64(1, 'D'))

        if custom_sims is None:
            price_simulations = self._price_model()
        else:
            price_simulations = custom_sims

        if fix_price is None:
            if custom_index is None:
                strikes = self._index_counter(price_simulations['futures'])
            else:
                strikes = custom_index
        else:
            strikes = np.resize(np.array([fix_price]), (n_sims, delivery_days))

        xes = []
        yes = []
        prems = []

        for i in range(10):

            prem = price_simulations['da'] * 0.1 * i
            lsmc_result = self._lsmc_counter(delivery_days=delivery_days,
                                             price_simulations=price_simulations['spot'],
                                             strikes=np.array(strikes), fee=prem)

            if prem == 0:
                prems.append(lsmc_result['premium'])
            else:
                pass
            if lsmc_result['option_cost'] < 0:

                xes.append(prem)
                yes.append(lsmc_result['option_cost'])

                break
            else:
                xes.append(prem)
                yes.append(lsmc_result['option_cost'])
            del (lsmc_result)

        premia = -(yes[-1] * (xes[-2] - xes[-1]) / (yes[-2] - yes[-1])) + xes[-1]

        # lsmc_result_0 = self._lsmc_counter(delivery_days=delivery_days, price_simulations=price_simulations['spot'],
        #                                   strikes=np.array(strikes), fee=premia)
        # print(lsmc_result_0['option_cost'])

        return {'err': np.std(prems) / (np.mean(prems) * (n_sims ** 0.5)) * 1.645, 'premium': premia, 'fee_0': yes[0]}

    def count_delta(self):

        price_range = [-60, -40, -30, -25, -20, -15, -12, -10, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8,
                       10, 12, 15, 20, 25, 30, 40, 60]
        delivery_days = int((self._delivery_end - self._delivery_start) / np.timedelta64(1, 'D'))

        diff_with_atm = []
        premia_with_fee0 = []
        for i in price_range:
            # print('i:', i)
            price_simulations = self._price_model(mult_for_delta=i)

            if self._fix_price is None:
                strikes = self._index_counter(price_simulations['futures'])
            else:
                strikes = np.resize(np.array([self._fix_price]), (self._n_sims, delivery_days))

            a = lsmc(max_acq=self._max_acq,
                     min_acq=self._min_acq,
                     max_dcq=self._max_dcq,
                     min_dcq=self._min_dcq,
                     delivery_days=delivery_days,
                     underlying_paths=price_simulations['spot'],
                     strike_paths=np.array(strikes),
                     premium=0)
            diff_with_atm.append(price_simulations['atm'] + i)
            premia_with_fee0.append(np.mean(a[0]))

        delta_der_values = []
        for i in diff_with_atm:
            delta_der_values.append(self._derivative_function(diff_with_atm, premia_with_fee0, i))

        return {'tcq':np.repeat(self._max_acq, len(delta_der_values)),
                'mtcq':np.repeat(self._min_acq, len(delta_der_values)),
                'dcq':np.repeat(self._max_dcq, len(delta_der_values)),
                'mdcq': np.repeat(self._min_dcq, len(delta_der_values)),
                'idx': np.repeat(str(self._indexes), len(delta_der_values)),
                'beg_date': np.repeat(self._delivery_start, len(delta_der_values)),
                'end_date': np.repeat(self._delivery_end, len(delta_der_values)),
                'diff_with_atm': diff_with_atm, 'premium': premia_with_fee0, 'delta_values': delta_der_values}

    def count_theta(self):

        delivery_days = int((self._delivery_end - self._delivery_start) / np.timedelta64(1, 'D'))

        diff_with_atm = []
        time_before_exp = (self._delivery_start - self._date_of_pricing) / np.timedelta64(1, 'D')
        step = np.floor(time_before_exp / 10)
        premia_with_fee0 = []
        date_and_time = []
        for i in range(0, int(time_before_exp)-30, int(step)):

            print('i:', i)
            date = self._date_of_pricing + np.timedelta64(i, 'D')

            price_simulations = self._price_model(mult_for_theta=i)

            if self._fix_price is None:
                strikes = self._index_counter(price_simulations['futures'], date,mult_for_theta=i)
            else:
                strikes = np.resize(np.array([self._fix_price]), (self._n_sims, delivery_days))


            a = lsmc(max_acq=self._max_acq,
                     min_acq=self._min_acq,
                     max_dcq=self._max_dcq,
                     min_dcq=self._min_dcq,
                     delivery_days=delivery_days,
                     underlying_paths=price_simulations['spot'],
                     strike_paths=np.array(strikes),
                     premium=0)

            diff_with_atm.append(i)
            premia_with_fee0.append(np.mean(a[0]))
            date_and_time.append(self._date_of_pricing + np.timedelta64(i,'D'))

        theta_der_values = []
        for i in diff_with_atm:
            theta_der_values.append(self._derivative_function(diff_with_atm, premia_with_fee0, i))

        return {'tcq':np.repeat(self._max_acq, len(theta_der_values)),
                'mtcq':np.repeat(self._min_acq, len(theta_der_values)),
                'dcq':np.repeat(self._max_dcq, len(theta_der_values)),
                'mdcq': np.repeat(self._min_dcq, len(theta_der_values)),
                'idx': np.repeat(str(self._indexes), len(theta_der_values)),
                'beg_date': np.repeat(self._delivery_start, len(theta_der_values)),
                'end_date': np.repeat(self._delivery_end, len(theta_der_values)),
                'time_from_sim_date': date_and_time, 'premium': premia_with_fee0, 'theta_values': theta_der_values}

    def count_vega(self):

        vol_range = [-0.5, -0.4, -0.3, -0.25, -0.2, -0.15, -0.1, -0.08, -0.07, -0.06, -0.05,
                     -0.04, -0.03, -0.02, -0.01,
                     0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

        delivery_days = int((self._delivery_end - self._delivery_start) / np.timedelta64(1, 'D'))

        diff_with_atm = []
        premia_with_fee0 = []
        for i in vol_range:
            # print('i:', i)
            price_simulations = self._price_model(mult_for_volat=i)

            if self._fix_price is None:
                strikes = self._index_counter(price_simulations['futures'], self._date_of_pricing)
            else:
                strikes = np.resize(np.array([self._fix_price]), (self._n_sims, delivery_days))

            a = lsmc(max_acq=self._max_acq,
                     min_acq=self._min_acq,
                     max_dcq=self._max_dcq,
                     min_dcq=self._min_dcq,
                     delivery_days=delivery_days,
                     underlying_paths=price_simulations['spot'],
                     strike_paths=np.array(strikes),
                     premium=0)
            diff_with_atm.append(i)
            premia_with_fee0.append(np.mean(a[0]))

        vega_der_values = []
        for i in diff_with_atm:
            vega_der_values.append(self._derivative_function(diff_with_atm, premia_with_fee0, i))

        return {'tcq':np.repeat(self._max_acq, len(vega_der_values)),
                'mtcq':np.repeat(self._min_acq, len(vega_der_values)),
                'dcq':np.repeat(self._max_dcq, len(vega_der_values)),
                'mdcq': np.repeat(self._min_dcq, len(vega_der_values)),
                'idx': np.repeat(str(self._indexes), len(vega_der_values)),
                'beg_date': np.repeat(self._delivery_start, len(vega_der_values)),
                'end_date': np.repeat(self._delivery_end, len(vega_der_values)),
                'vol_delta': diff_with_atm, 'premium': premia_with_fee0, 'vega_values': vega_der_values}

    def count_shadow_greek(self, n_sims: int = 2500, custom_sims: np.array = None, custom_index: np.array = None,
                           fix_price: float = None):

        """
        Считает гибкость и подгоняет премию.

        Attributes:
        ----------
        max_acq
            Максимальный суммарный отбор по контракту
        min_aqc
            Минимальный суммарный отбор по контракту
        max_dcq
            Максимальный суточный отбор по контракту
        min_dcq
            Минимальный суточный отбор по контракту
        n_sims: int = 5000, optional
            Количество симуляций
            БОЛЬШЕ 7500 НЕ СТАВИТЬ!!! РАСЧЕТЫ ПАДАЮТ.
        custom_sims: np.array = None

        custom_index: np.array = None

        fix_price:float=None
        """

        delivery_days = int((self._delivery_end - self._delivery_start) / np.timedelta64(1, 'D'))

        if custom_sims is None:
            price_simulations = self._price_model()
        else:
            price_simulations = custom_sims

        if fix_price is None:
            if custom_index is None:
                strikes = self._index_counter(price_simulations['futures'])
            else:
                strikes = custom_index
        else:
            strikes = np.resize(np.array([fix_price]), (n_sims, delivery_days))

        fee_0 = []
        maq_pc = []
        premium = []
        for j in range(20):
            xes = []
            yes = []
            prems = []
            print(j)
            for i in range(10):

                prem = price_simulations['da'] * 0.1 * i
                lsmc_result = self._lsmc_counter(delivery_days=delivery_days,
                                                 price_simulations=price_simulations['spot'],
                                                 strikes=np.array(strikes), fee=prem,mult_for_greek=j*0.05)

                if prem == 0:
                    prems.append(lsmc_result['option_cost'])
                else:
                    pass
                if lsmc_result['option_cost'] < 0:

                    xes.append(prem)
                    yes.append(lsmc_result['option_cost'])

                    break
                else:
                    xes.append(prem)
                    yes.append(lsmc_result['option_cost'])
                del (lsmc_result)

            premia = -(yes[-1] * (xes[-2] - xes[-1]) / (yes[-2] - yes[-1])) + xes[-1]

            # lsmc_result_0 = self._lsmc_counter(delivery_days=delivery_days, price_simulations=price_simulations['spot'],
            #                                   strikes=np.array(strikes), fee=premia)
            # print(lsmc_result_0['option_cost'])
            fee_0.append(yes[0])
            maq_pc.append(j)
            premium.append(premia)

        return {'tcq':np.repeat(self._max_acq, len(fee_0)),
                'mtcq':np.repeat(self._min_acq, len(fee_0)),
                'dcq':np.repeat(self._max_dcq, len(fee_0)),
                'mdcq': np.repeat(self._min_dcq, len(fee_0)),
                'idx': np.repeat(str(self._indexes), len(fee_0)),
                'beg_date': np.repeat(self._delivery_start, len(fee_0)),
                'end_date': np.repeat(self._delivery_end, len(fee_0)),
                'premium': premium, 'maq_pc': maq_pc, 'fee_0': fee_0}


if __name__ == '__main__':
    t_0 = time.time()
    contract = VolumeFlex(delivery_start=np.datetime64('2023-01-01'), delivery_end=np.datetime64('2023-04-01'),
                          delivery_point='TTF',
                          date_of_pricing=np.datetime64('2022-05-18'), date_of_volat=np.datetime64('2022-05-17'),
                          indexes=[1, 0, 1], max_acq=100 * 90, min_aqc=10 * 90, max_dcq=105,
                          min_dcq=0,
                          n_sims=2500)
    print(contract.count_premium())
    print(time.time() - t_0)
    delta = contract.count_delta()
    print(delta)
    pd.DataFrame.from_dict(delta).to_excel('/mnt/teamdocs_ns/TRD_Exchange/Valuations/Greeks_test/cdelta.xlsx',index=False)
    print(time.time() - t_0)
    theta = contract.count_theta()
    print(theta)
    pd.DataFrame.from_dict(theta).to_excel('/mnt/teamdocs_ns/TRD_Exchange/Valuations/Greeks_test/ctheta.xlsx',index=False)
    print(time.time() - t_0)
    vega = contract.count_vega()
    print(vega)
    pd.DataFrame.from_dict(vega).to_excel('/mnt/teamdocs_ns/TRD_Exchange/Valuations/Greeks_test/cvega.xlsx',index=False)
    print(time.time() - t_0)
    sg = contract.count_shadow_greek()
    print(sg)
    pd.DataFrame.from_dict(sg).to_excel('/mnt/teamdocs_ns/TRD_Exchange/Valuations/Greeks_test/maq_greek.xlsx',index=False)
    print(time.time() - t_0)

