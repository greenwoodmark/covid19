# -*- coding: utf-8 -*-
"""
derived from COVID19.ipynb
"""

import pandas as pd
import numpy as np


from scipy.stats import nbinom

#---------------------------------------------------------------------------------
def fit_err(params, hyperparams):
    s,p,n = params
    df = hyperparams
    
    if 'lockdown' in df.columns:
        df = df.loc[df.index>=lockdown_date]   #we only fit error after this date
    
    #proj_df = pd.DataFrame(index=df.index)
    
    #update the model projection this set of parameters
    days_limit = df.index.shape[0]  #we limit this to prevent the fitter taking too long
    arr = np.array(range(days_limit))  #we limit analysis to days_limit after infection
    nbpr = lambda x: nbinom.pmf(x, n, p)
    negbin_probabilities = nbpr(arr)
    
    #pad the negbin_probabilities array with 0 in front
    #np.pad(a, (1, 0), 'constant', constant_values=(0, 0))
    num_rows = df.index.shape[0]
    num_cols = negbin_probabilities.shape[0]+num_rows
    pmatrix = np.zeros((num_rows,num_cols))
    for m in range(num_rows):
        front_pad_cols = m
        back_pad_cols = num_rows-m
        pmatrix[m,:] = np.pad(negbin_probabilities, (front_pad_cols, back_pad_cols), 'constant', constant_values=(0, 0))
        
    nmd_prob = np.matmul(df['new_cases'].to_numpy().reshape(1,-1).astype(float),pmatrix.astype(float))
    new_model_deaths = nmd_prob[:,:num_rows] * (1-s)
    new_deaths = df['new_deaths'].values.reshape(1,-1).astype(float)    
    error_squared = (new_deaths - new_model_deaths)**2 
    latest_error_squared = error_squared.sum()
    
    return latest_error_squared

#-------------------------------------------------
#define the optimiser associated with our fit function
def fit_model(initparams, hyperparams, bounds, maxiter):
    from scipy import optimize as opt
    #res  = opt.fmin_slsqp(fit_err, initparams, args=(hyperparams,), bounds = bounds, iter=30)
    res  = opt.fmin_l_bfgs_b(fit_err, initparams, args=(hyperparams,), approx_grad =True, bounds = bounds, maxiter = maxiter)
    return res        
#-------------------------------------------------

#define our fit function and the associated optimiser
#TODO consider seasonality

   
#-------------------------------------------------
def projection_df(params, df, cases_growth_rate=0):
    """
    plot model projection for last set of parameters
    extends df 100 days into the future and derives new deaths based on parameters in params 
    and new cases growth parameter rate cases_growth_rate
    """
    s,p,n = params
    if 'lockdown' in df.columns:
        df = df.loc[df.index>=lockdown_date]   #we only fit error after this date

    next_datetime = df.index[-1]+pd.DateOffset(1) 
    index_list = list(df.index)+list(pd.date_range(next_datetime, periods=100, freq='D').values)
    index_list = [pd.Timestamp(d) for d in index_list]
    #reindex df to index_list
    df = df.reindex(index_list)
    df['cases']=df['cases'].fillna(method='ffill')
    df['deaths']=df['deaths'].fillna(method='ffill')
    df = df.fillna(0.0)
    
    days_limit = df.index.shape[0]  #we limit this to prevent the fitter taking too long
    arr = np.array(range(days_limit))  #we limit analysis to days_limit after infection
    nbpr = lambda x: nbinom.pmf(x, n, p)
    negbin_probabilities = nbpr(arr)

    print('the negative binomial model projected for', str(days_limit),'days accounts for',
      round(100*negbin_probabilities[0:days_limit].sum(),3),'% of future deaths')
    
    num_rows = days_limit
    num_cols = negbin_probabilities.shape[0]+num_rows
    pmatrix = np.zeros((num_rows,num_cols))
    for m in range(num_rows):
        front_pad_cols = m
        back_pad_cols = num_rows-m
        pmatrix[m,:] = np.pad(negbin_probabilities, (front_pad_cols, back_pad_cols), 'constant', constant_values=(0, 0))
        
    nmd_prob = np.matmul(df['new_cases'].to_numpy().reshape(1,-1).astype(float),pmatrix.astype(float))
    new_model_deaths = nmd_prob[:,:num_rows] * (1-s)
    df['new_model_deaths'] = new_model_deaths.squeeze()
    
    return df
#-------------------------------------------------


#________________________________________________________________________________
if __name__ == "__main__":
    #country='Spain'; lockdown_date = '2020-03-15'
    #country='Italy'; lockdown_date = '2020-03-09'
    country='United Kingdom';  #'2020-03-23'
    #country='France'
    #country='Switzerland'
    #country='China'
    #country='US'; lockdown_date = None #'2020-03-22' #NY date
    lockdown_date = None
    
    #confirmed cases in time_series_covid19_confirmed_global.csv
    url_c = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    file_c = 'C:/Users/Mark/Documents/Python/code/time_series_covid19_confirmed_global.csv'
    read_c = file_c #url_c or local file_c if saved already
    if (read_c==file_c):
        print()
        print('NB: TOMORROW REMEMBER TO TURN SOURCE BACK TO URL FROM LOCAL')
        print()
    df_c = pd.read_csv(read_c)   #global confirmed cases
    df_c['Province/State'] = df_c['Province/State'].fillna('ALL')
    df_cc = df_c.loc[(df_c['Country/Region']==country) & (df_c['Province/State']=='ALL')]
    df_cc = df_cc.T; df_cc.columns=['cases']
    df_cc = df_cc.iloc[4:]
    df_cc.index = pd.to_datetime(df_cc.index)
    
    #deaths in time_series_covid19_deaths_global.csv
    url_d = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    file_d = 'C:/Users/Mark/Documents/Python/code/time_series_covid19_deaths_global.csv'
    if (read_c==url_c):
        read_d = url_d #url_d or local file_d if saved already
    else:
        read_d = file_d
    df_d = pd.read_csv(read_d)   #global confirmed cases
    df_d['Province/State'] = df_d['Province/State'].fillna('ALL')
    df_dc = df_d.loc[(df_d['Country/Region']==country) & (df_d['Province/State']=='ALL')]
    df_dc = df_dc.T; df_dc.columns=['deaths']
    df_dc = df_dc.iloc[4:]
    df_dc.index = pd.to_datetime(df_dc.index)
    
    df = pd.concat([df_cc,df_dc], axis=1, sort=False)
    df['country'] = country

    if pd.isnull(lockdown_date):
        df = df[['country','cases','deaths']]
    else:
        df['lockdown'] = lockdown_date    
        df = df[['country','lockdown','cases','deaths']]        
    
    df['new_deaths'] = df['deaths']-df['deaths'].shift(1)
    df['new_cases'] = df['cases']-df['cases'].shift(1)
    df.at[df.index[0],'new_deaths']=df.loc[df.index[0],'deaths']
    df.at[df.index[0],'new_cases']=df.loc[df.index[0],'cases']
    df['new_cases_rate'] = df['new_cases'] / (0.5*df['cases'].shift(1)+0.5*df['cases']+1e-10)
    df.at[df.index[0],'new_cases_rate']=0.0
    
    df.tail()
    
    #df['new_cases_rate'].plot()
    
    df['new_cases_rate'].iloc[-50:].plot(title=country+' new_cases_rate', ylim=(0,0.5))
    #df['new_cases_rate'].plot(logy=True,title=country+' new_cases_rate (log scale)')
    
    #look at variability in last 5 days and extend back as far as say double this variability
    #look at variability in last 5 days and extend back as far as say double this variability
    #look at variability in last 5 days and extend back as far as say double this variability
    df['new_cases_rate'].iloc[-5:].std()
    df['new_cases_rate'].iloc[-23:-5].std()
    #choose 
 


    #===================================================================fit each day lockdown+6d
    bounds_tuple = ((0.1,0.99),(0.1,0.9),(1.0,100.0))   #s, p, n
    max_iterations = 50
    init_params_tuple = (0.5,0.35,7.0)
    
    if pd.isnull(lockdown_date):
        start_loc = 30 #pointless fitting 3 parameters to less than 30 data points
        end_loc = df.shape[0]
    else:    
        lockdown_loc = df.index.get_loc(lockdown_date)
        start_loc = lockdown_loc + 6
        end_loc = df.shape[0]
    
    for i in range(start_loc, end_loc):
        fit_df = df.iloc[:i]
        res=fit_model(initparams=init_params_tuple, hyperparams=fit_df, bounds=bounds_tuple, maxiter=max_iterations)
        s,p,n=res[0]
        mean = nbinom.mean(n, p) #n*p/(1-p)
        print(df.index[i].strftime('%Y-%m-%d'),'parameters=',res[0],',',round(mean,1),'days to death,', round(res[1],0),'error')
        init_params_tuple = res[0]
        s,p,n = res[0]
        df.at[df.index[i],'s']=s
        df.at[df.index[i],'p']=p
        df.at[df.index[i],'n']=n
    #===================================================================
    

    proj_df = projection_df(params=res[0], df=df, cases_growth_rate=0.1)
    
    proj_df['new_deaths'] = proj_df['new_deaths'].replace(0.0, np.nan) #don't plot zero values
    
    s,p,n = res[0]
    #mean = n*(1/p - 1) = n*(1-p)/p ##C:\ProgramData\Anaconda3\Lib\site-packages\scipy\stats\_discrete_distns.py line 210
    mean = nbinom.mean(n, p) 
    print('mean days to death for those who do not survive =',round(mean,1),'days')
    
    proj_df[['new_deaths','new_model_deaths']].iloc[30:-60].plot(title=country+' fitted model, log scale for y-axis', logy=True)
    proj_df[['new_deaths','new_model_deaths']].iloc[30:-60].plot(title=country+' fitted model')

    loc_max = proj_df.loc[ proj_df['new_model_deaths'] == proj_df['new_model_deaths'].max()].index[0]
    print(int(round(proj_df['new_model_deaths'].max(),0)),'max deaths on',loc_max.strftime('%Y-%m-%d'))

    proj_df['model_new_deaths_rate'] = proj_df['new_model_deaths']/proj_df['deaths']
    print(proj_df.tail(115).head(15))
    proj_df['model_new_deaths_rate'].iloc[30:-60].plot(title=country+' model new deaths rate')

    print()
    '''
    Negative binomial distribution describes a sequence of i.i.d. Bernoulli trials, 
    repeated until a predefined, non-random number of successes occurs.
    X distributed negative binomial (n,p) gives number of failures until n-th success (so always x+n trials) 
    '''
    df.to_clipboard()