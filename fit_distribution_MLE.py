#Import important libraries
import numpy as np
import scipy.stats as stats


#Define a function to find best t-distribution that fits the data
#Find degrees of freedom with maximum liklihood estimate
#Return estimated dof & cdf for the time-series
def fit_tdist_logMLE(input_ts, dof):
   
   #Number of risk factors
   num_rf=input_ts.shape[0]
   len_ts=input_ts.shape[1]
    
   #Output array to store log LE by dof and fitted dof
   mle_dof=np.zeros((num_rf,len(dof)))
   fitted_nu=np.zeros(num_rf) 

   #Construct time flexible probabilities
   fp=flex_prob(len_ts/2,len_ts)

   #For each return series, construct MLE arrays 
   for j in np.arange(0,num_rf):
        ret_ts=input_ts[:][j]
        fp_mean=np.dot(ret_ts,fp)
        fp_var=np.dot((ret_ts-fp_mean)**2, fp)
        print(fp_mean)
        print(np.sqrt(fp_var))
                      
        #For each degree of freedom, find log LE   
        for k in np.arange(0, len(dof)):
           if(dof[k] > 2): 
             t_dist=stats.t(dof[k], loc=fp_mean, scale=np.sqrt((dof[k]-2)/dof[k])*fp_var) 
           else:
             t_dist=stats.t(dof[k], loc=fp_mean)
           mle_dof[j][k]=np.sum([t_dist.logpdf(ret) for ret in ret_ts])
     
        #Find max LE 
        max_idx=np.argsort(mle_dof[j])[::-1][0]
        fitted_nu[j]=dof[max_idx]
   
   return fitted_nu


#Returns cdf based on the input degrees of freedom
def get_cdf_tdist(input_ts, nu):
            
     #Output array for uniform variates
     uniform_variates=np.zeros((input_ts.shape[0], input_ts.shape[1]))   
    
     #Loop through the risk factors
     for j in np.arange(0, input_ts.shape[0]):
        #Create distribution with given dof
        fitted_t_dist=stats.t(nu[j]) 
        uniform_variates[j][:]=fitted_t_dist.cdf(input_ts[j][:])
    
     return uniform_variates


#Function to get flexible probabilities
def flex_prob(half_life, num_days):
    
    time_flex_prob= np.exp((-np.log(2)/half_life)*np.arange(num_days,0,-1))
    scale_to_one=1/np.sum(time_flex_prob)
     
    return scale_to_one*time_flex_prob



#Generate any distribution invariants using uniform residuals
def get_inverse_cdf(uniform_variates):
    
     #Initiate matrix of t-distribution residuals with dof=4
     t_dof4_variates=np.zeros((uniform_variates.shape[0], uniform_variates.shape[1]))
    
     for count in np.arange(0, uniform_variates.shape[0]):
        #Get t-distribution invariants with dof=4
        t_dist_df4=stats.t(4)
        t_dof4_variates[count][:]=t_dist_df4.ppf(uniform_variates[count])
        
     return t_dof4_variates




    