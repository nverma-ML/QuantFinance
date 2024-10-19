#Import python libraries
import numpy as np
import scipy.stats as stats


class tCopula:
    
    dof=4 #default degrees of freedom defined 
    
    def __init__(self, corrMatrix):
        self.corrMatrix=corrMatrix
                   
    #Generate random variables based on t-distribution with default dof
    def generate_random_variates(self, num_ts, num_scenarios):
        random_t_variates=stats.multivariate_t.rvs(loc=np.zeros(num_ts), shape=self.corrMatrix, df=self.dof, size=num_scenarios)
        return random_t_variates