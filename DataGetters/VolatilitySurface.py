import pandas as pd
import py_vollib_vectorized
from sqlalchemy import create_engine
from datetime import datetime
import scipy as sp
import numpy as np
import pdb

class IceVolat:

    def __init__(self,date_of_option:str,options_type:str='Put',table_type:str='atm'):

        # options_type can be Put, Call or all
        # table_type can be all, or atm
        self.date_of_option = date_of_option
        self.table_type = table_type
        if options_type=='all':
            data = self.get_ice_data_from_base('Put')
            data1 = self.get_ice_data_from_base('Call')
            self.all_data = pd.concat([data,data1],axis=0)
        else:
            self.all_data = self.get_ice_data_from_base(options_type)

    def get_ice_data_from_base(self,options_type:str):
        engine = create_engine('postgresql://postgres:!QAZ2wsx@10.18.22.217/analytics_base', echo=False)
        options = pd.read_sql(f"SELECT * FROM f_get_all_options('{options_type}','{self.date_of_option}')", con=engine)
        options = options.replace({'Put':'p','Call':'c'})
        time_delta = []
        for row in options.to_numpy():
            #print(row)
            time_delta.append(((datetime.strptime(str(row[4])[:10],'%Y-%m-%d')-datetime.strptime(str(row[0])[:10],'%Y-%m-%d')).days+15)/365)
        options['t']=time_delta
        return options

    def get_volatility(self,ice_data:pd.DataFrame=None):
        if ice_data is None:
            options = self.get_ice_data_from_base(self.date_of_option)
        else:
            options = ice_data

        options['imvol'] = py_vollib_vectorized.vectorized_implied_volatility(
            price=options['premium'],
            S = options['value'],
            K = options['strike'],
            t = options['t'],
            r=0,
            flag=options['option_type'],
            q=0,
            return_as='numpy',on_error='ignore')

        options = options.dropna()
        #pdb.set_trace()
        options = options.set_index('beg_date').join(options.groupby('beg_date').size().rename('counts'))
        options = options[options['counts']>2]
        return options

    def interpolation(self, ice_data:pd.DataFrame=None):
        if ice_data is None:
            options = self.get_volatility(self.all_data)
        else:
            options = ice_data
        options = options.sort_values(by=['beg_date', 'strike'])
        options = options.drop_duplicates(subset=['product_code', 'strike'], keep='last')
        interpolation_functions = {pd.Timestamp(month).strftime('%Y-%m-%d'):
            sp.interpolate.CubicSpline(
                options[(options.index == month)]['strike'],
                options[(options.index == month)]['imvol'],
                bc_type='not-a-knot')
            for month in options.index.unique()}
        return interpolation_functions

    def get_table(self,interpolation_functions:dict=None,ice_data:pd.DataFrame=None):
        if ice_data is None:
            options = self.get_volatility(self.all_data)
            interpolation_functions = self.interpolation()
        else:
            options = ice_data
        options = options.sort_values(by=['beg_date', 'strike'])
        options = options.drop_duplicates(subset=['product_code'], keep='last')

        #pdb.set_trace()
        data = pd.DataFrame(
            np.transpose(list(
                map(lambda x: interpolation_functions[str(
                    list(interpolation_functions.keys())[x])[:10]](
                    options['value'].to_numpy()[x] * np.array(list(
                        map(lambda x: 0.5 + 0.01 * x, range(101))))), range(len(list(interpolation_functions.keys())))))),
            columns=list(interpolation_functions.keys()),
            index=list(map(lambda x: 0.5 + 0.01 * x, range(101))))
        if self.table_type=='atm':
            data = data[data.index==1]
        else:
            pass
        return data

if __name__ =='__main__':
    print(IceVolat('2022-06-13','all','atm').get_table())

