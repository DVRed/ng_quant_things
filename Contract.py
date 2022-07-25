from numpy import datetime64

class Contract:
    
    def __init__(self,beg_date:datetime64,end_date:datetime64,dcq:int,**other_params):
        
        self.beg_date = beg_date
        self.end_date = end_date
        self.dcq = dcq
        for param,value in other_params.items():
            if str(param) == 'min_dcq' and type(value)==int:
                self.min_dcq = value
            elif str(param) == 'min_tcq' and type(value)==int:
                self.min_tcq = value
            elif str(param) == 'max_tcq' and type(value)==int:
                self.max_tcq = value
            elif str(param) == 'indexes' and type(value)==list:
                self.indexes = value
            else:
                raise ValueError('unknown parameter')


