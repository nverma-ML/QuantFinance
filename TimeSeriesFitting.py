from arch import arch_model
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

class __FinTimeSeries__ :
                     
    def __init__(self, fin_time_series):
        self.fin_time_series=fin_time_series
        
    def fitGARCH(self):
        garch_model = arch_model(self.fin_time_series, vol='GARCH', p=1, q=1)
        self.garch_fit=garch_model.fit()
        self.conditional_vol=self.garch_fit.conditional_volatility
        self.main_residuals=self.garch_fit.resid
        
    def fitARModel(self, lag):
        arModel = AutoReg(self.fin_time_serie, lags=lag)
        arModel.fit()
        arModel.ar_lags
              
    def get_garch_params(self):
        return self.garch_fit.params
            
    def get_conditional_vol(self):
        return self.conditional_vol
    
    def get_main_residuals(self):
        return self.main_residuals

    def get_iid_residuals(self):
        return np.divide(self.main_residuals, self.conditional_vol)
    
    def print_ts(self):
        print(self.fin_time_series)