import pdb
import numpy as np
import pandas as pd
import math
from sqlalchemy import create_engine
import time


class IndexCounter:

    def __init__(self):

        """
        Класс для расчёта индексов, которые можно записать в формате (x,y,z), где
        x - количество инструментов берем
        y - лаг
        z - количество месяцев усреднения.
        Данные по частично сформированным индексам подтягиваются из аналитической базы ССТД.
        Источник: ICE, settlement price.

        Methods
        ----------
        _get_data_from_base(from_contract:np.datetime64, from_date:np.datetime64, to_date:np.datetime64, hub:str='TTF', n_instruments: int = 36)
            Забирает данные из базы аналитики ССТД
        _data_cleaner(self, simulation_date: np.datetime64, real_data: pd.DataFrame = None, start_date: np.datetime64 = None)
            Использование вне класса не предусмотрено
        _contract_start_definer(self, ticker: str)
            Использование вне класса не предусмотрено. Выглядит, как бесполезная и неиспользуемая функция.
        _prices_concater(self, data_real, data_sim)
            Конкатит цены реальные и симулированные. Использование вне класса не предусмотрено.
        index_count(self, contract_start, contract_end, simulation_date, indexes, data)
            Основная функция в классе. Считает индекс.
        """

        pass

    def _get_data_from_base(self, from_contract: np.datetime64, from_date: np.datetime64, to_date: np.datetime64,
                            hub: str = 'TTF', n_instruments: int = 36):

        engine = create_engine('postgresql://postgres:!QAZ2wsx@10.18.22.217/analytics_base', echo=False)
        from_contract_m = str(from_contract)[:10]
        from_date_m = str(from_date)[:10]
        to_date_m = str(to_date)[:10]
        futures = pd.read_sql(
            f"SELECT * FROM f_get_all_forwards('{from_date_m}','{to_date_m}','{hub}','{from_contract_m}')",
            con=engine)

        return futures

    def _data_cleaner(self, simulation_date: np.datetime64, real_data: pd.DataFrame = None,
                      start_date: np.datetime64 = None):
        date_index = list(
            map(
                lambda x: start_date + np.timedelta64(x, 'D'),
                range(
                    int((simulation_date - start_date) / np.timedelta64(1, 'D'))
                )
            )
        )
        return pd.concat([pd.DataFrame(index=date_index), real_data], axis=1).sort_index()

    def _contract_start_definer(self, ticker: str):

        month = {'F': '01', 'G': '02', 'H': '03', 'J': '04', 'K': '05', 'M': '06',
                 'N': '07', 'Q': '08', 'U': '09', 'V': '10', 'X': '11', 'Z': '12'}
        year = {'2': '2022', '3': '2023', '4': '2024',
                '5': '2025', '6': '2026', '7': '2027',
                '8': '2028', '9': '2029', '0': '2030'}
        t_year = ticker.split(' ')[0][-1]
        t_month = ticker.split(' ')[0][-2]
        return np.datetime64(year[t_year] + '-' + month[t_month], 'D')

    def _prices_concater(self, data_real, data_sim):

        # # цены различных продуктов
        bloom_tickers_pivot = data_real.reset_index(drop=True).to_numpy()

        # # повторил предыдущий массив столько раз, сколько симуляций проделано в data_sim
        bloom_tickers_pivot_repeated = np.tile(bloom_tickers_pivot, (data_sim.shape[0], 1, 1))

        # # в каждой симуляции слепил реальные данные с симуляциями
        bloom_tickers_pivot_repeated = bloom_tickers_pivot_repeated[:, :, :data_sim.shape[-1]]
        real_data_x_simulations = np.append(bloom_tickers_pivot_repeated,
                                            data_sim[:, :, :bloom_tickers_pivot_repeated.shape[-1]], axis=1)
        return real_data_x_simulations

    def index_count(self, contract_start: np.datetime64, contract_end: np.datetime64, simulation_date: np.datetime64,
                    indexes: list, data: np.array,mult_for_theta:int=0):
        """
        Функция для расчёта индекса, который можно записать в формате (x,y,z)

        Attributes
        ----------
        contract_start:np.datetime64
            Дата начала поставки по контракту
        contract_end:np.datetime64
            Дата окончания поставки по контракту
        simulation_date:np.datetime64
            Дата проведения ценовых симуляций
        indexes:list
            Принимает список только целых значений
            Индексы (страйки) в формате:
            1 - сколько инструментов берем
            2 - лаг
            3 - сколько месяцев усреднения
            4 - сколько месяцев действует индекс (можно скипнуть)
            Пример:indexes = [1,0,1,1] - классический Month-Ahead Index
        data:np.array
            Ценовые симуляции(выходные просимулированы, как обычные дни)
        """

        if len(indexes) == 3:
            indexes.append(indexes[0])
        else:
            pass

        # дата начала расчетов
        start_count = np.datetime64(str(contract_start)[:4], 'Y') + np.timedelta64(int(str(contract_start)[5:7]) - 1,
                                                                                   'M') - np.timedelta64(
            indexes[2] + indexes[1], 'M') + np.timedelta64(0, 'D')


        # get fully concated data and missing is np.nan
        if simulation_date > start_count:

            month_delta = int((np.datetime64(simulation_date,'M')-np.datetime64(simulation_date-np.timedelta64(mult_for_theta,'D'),'M'))/np.timedelta64(1,'M'))


            # change ticker as tztz* to np.datetime as delivery start date
            real_data = self._get_data_from_base(
                contract_start,
                np.datetime64(start_count,'M')-np.timedelta64(month_delta,'M')+np.timedelta64(int(str(start_count)[-2:])-1,'D'),
                np.datetime64(simulation_date,'M')-np.timedelta64(month_delta,'M')+np.timedelta64(int(str(simulation_date)[-2:])-1,'D'))

            # sort by delivery start date and date of value
            real_data = real_data.sort_values(['beg_date', 'date']).reset_index()
            real_data = real_data.pivot(
                index='date',
                columns='beg_date',
                values='value'
            )

            clean_real_data = self._data_cleaner(simulation_date, real_data, start_count)
            concated_prices = self._prices_concater(clean_real_data, data)
            simulation_date = start_count
        else:
            concated_prices = data

        data = concated_prices
        # месяца для расчетов
        month = list(map(lambda x: int(str(np.datetime64(str(contract_start)[:7]) + np.timedelta64(x, 'M'))[5:7]),
                         range(int((np.datetime64(str(contract_end)[:7]) - np.datetime64(
                             str(contract_start)[:7])) / np.timedelta64(1, 'M')))))

        # Даты в которые действует индекс
        days_of_index = list(map(
            lambda x: np.datetime64(contract_start, 'M') + np.timedelta64(indexes[-1] * x, 'M') + np.timedelta64(0,
                                                                                                                 'D'),
            range(math.ceil(len(month) / indexes[-1]) + 1)))

        # дни для функции numpy repeat
        days_of_index_repeated = list(map(lambda x: (days_of_index[x + 1] - days_of_index[x]) / np.timedelta64(1, 'D'),
                                          range(len(days_of_index) - 1)))

        # даты начала и конца каждого индекса
        days_index_count = list(map(lambda x: [
            np.datetime64(days_of_index[x], 'M') - np.timedelta64(indexes[1] + indexes[2], 'M') + np.timedelta64(0,
                                                                                                                 'D'),
            np.datetime64(days_of_index[x], 'M') - np.timedelta64(indexes[1], 'M') + np.timedelta64(0, 'D')],
                                    range(len(days_of_index) - 1)))

        # финальные расчеты
        final_index = list(map(lambda y: np.concatenate(
            list(
                map(
                    lambda x: np.repeat(
                        np.nanmean(
                            data[y][int((days_index_count[x][0] - simulation_date) / np.timedelta64(1, 'D')):int(
                                (days_index_count[x][1] - simulation_date) / np.timedelta64(1, 'D')),
                            indexes[0] * (x):indexes[0] * (x + 1)]), days_of_index_repeated[x]),
                    range(len(days_index_count))))),
                               range(len(data))))
        return final_index


if __name__ == '__main__':
    from Models.price_model import PriceModel

    # sim = np.load('/srv/sstd/Projects/TRD_SSTD/Workspace/d_martynov/Models/lsmc/sims/sim_TTF_2022-01-07.npz')
    # ttf_frd = sim['sim.frd']

    prices = PriceModel('simple', 2, np.datetime64('2021-01-01'), np.datetime64('2022-01-01'),
                        np.datetime64('2021-12-25'), 'TTF').simple_model()

    sim = pd.HDFStore('/home/d_devyatkin/Simulations.h5', mode='r')
    t_0 = time.time()
    c = IndexCounter().index_count(
        np.datetime64('2021-01-01'),
        np.datetime64('2022-01-01'),
        np.datetime64('2022-01-02'),
        [3, 0, 1],
        prices['futures'])
    print(c)


