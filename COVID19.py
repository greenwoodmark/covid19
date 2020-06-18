# -*- coding: utf-8 -*-
"""
module to support COVID19.ipynb notebook to fit negative binomial model to
Johns Hopkins COVID-19 cases and deaths data by country
"""

import pandas as pd
import numpy as np
from scipy.stats import nbinom
import matplotlib.pyplot as plt
import multiprocessing

#---------------------------------------------------------------------------------
def ew_halflife(n, halflife):
    """
    returns np array of exponential weights given halflife, see pandas.ewma docs
    """
    alpha = 1-np.exp(np.log(0.5)/halflife)
    ewalpha = lambda x: (1-alpha)**(n-x)
    weights = ewalpha(np.arange(1, n+1))
    return weights


#---------------------------------------------------------------------------------
def save_data(localpath='C:/Users/Mark/Documents/Python/code/covid19/'):
    from urllib.request import urlretrieve
    url_c = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    urlretrieve(url_c, localpath+'/time_series_covid19_confirmed_global.csv')
    url_d = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    urlretrieve(url_d,localpath+'/time_series_covid19_deaths_global.csv')
    return


#---------------------------------------------------------------------------------
def investigate_data():
    """
    return list of countries in dataset
    """    
    #confirmed cases in time_series_covid19_confirmed_global.csv
    url_c = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    df_c = pd.read_csv(url_c)   #global confirmed cases
    country_list = list(set(df_c['Country/Region']))
    country_list.sort() 
    return country_list

#---------------------------------------------------------------------------------
def prepare_data(country='United Kingdom', lockdown_date = None, URLnotfile = True):
    """
    extract Johns Hopkins COVID-19 cases and deaths data by country from github urls
    
    return df, a pandas DataFrame of containing the data
    
    Notes
    =====
    To use local CSV files for data, 
        1) use URLnotfile = False
        2) set file_c, file_d below for path and name of cases and deaths data

    """    

    #confirmed cases in time_series_covid19_confirmed_global.csv
    url_c = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    file_c = 'C:/Users/Mark/Documents/Python/code/covid19/time_series_covid19_confirmed_global.csv'
    
    if URLnotfile:
        read_c = url_c #url_c or local file_c if saved already
    else:
        read_c = file_c #url_c or local file_c if saved already
        
    df_c = pd.read_csv(read_c)   #global confirmed cases
    df_c['Province/State'] = df_c['Province/State'].fillna('ALL')
    if country=='China':
        df_cc = df_c.loc[(df_c['Country/Region']==country)].sum(axis=0)
        df_cc = pd.DataFrame(df_cc).T
    else:
        df_cc = df_c.loc[(df_c['Country/Region']==country) & (df_c['Province/State']=='ALL')]
    df_cc = df_cc.T; df_cc.columns=['cases']
    df_cc = df_cc.iloc[4:]
    df_cc.index = pd.to_datetime(df_cc.index)
    
    #deaths in time_series_covid19_deaths_global.csv
    url_d = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    file_d = 'C:/Users/Mark/Documents/Python/code/covid19/time_series_covid19_deaths_global.csv'
    if (read_c==url_c):
        read_d = url_d #url_d or local file_d if saved already
    else:
        read_d = file_d
    df_d = pd.read_csv(read_d)   #global confirmed cases
    df_d['Province/State'] = df_d['Province/State'].fillna('ALL')
    if country=='China':
        df_dc = df_d.loc[(df_d['Country/Region']==country)].sum(axis=0)
        df_dc = pd.DataFrame(df_dc).T
    else:
        df_dc = df_d.loc[(df_d['Country/Region']==country) & (df_d['Province/State']=='ALL')]
    df_dc = df_dc.T; df_dc.columns=['deaths']
    df_dc = df_dc.iloc[4:]
    df_dc.index = pd.to_datetime(df_dc.index)
   
    # 2020-04-24 are some countries' new cases net of recoveries?
    # e.g. France has negative new cases    
    
    df = pd.concat([df_cc,df_dc], axis=1, sort=False)
    
    df['country'] = country
    df['latest_data_date'] = df_dc.index.max().strftime('%Y-%m-%d')
    
    if pd.isnull(lockdown_date):
        df = df[['country','latest_data_date','cases','deaths']]
    else:
        df['lockdown'] = lockdown_date    
        df = df[['country','latest_data_date','lockdown','cases','deaths']]
    
    df['new_deaths'] = df['deaths']-df['deaths'].shift(1)
    df['new_cases'] = df['cases']-df['cases'].shift(1)
    df.at[df.index[0],'new_deaths']=df.loc[df.index[0],'deaths']
    df.at[df.index[0],'new_cases']=df.loc[df.index[0],'cases']
    df['new_cases_rate'] = df['new_cases'] / (df['cases'].shift(1)+1e-10)
    df.at[df.index[0],'new_cases_rate']=0.0
 
    negative_rates_list = list(df.loc[df['new_cases_rate']<0].index)
    negative_rates_list = [d.strftime('%Y-%m-%d') for d in negative_rates_list]
    if negative_rates_list: #warn
        print(); print('--------------------------------------------------------------------')
        print('WARNING: check negative growth in new cases for dates',negative_rates_list)
        print('         -> new cases and growth rates for these dates have been set to zero')
        print(); print('--------------------------------------------------------------------')
        df.at[df['new_cases']<0,'new_cases']=0.0        
        df.at[df['new_cases_rate']<0,'new_cases_rate']=0.0   
    
    return df

#---------------------------------------------------------------------------------
def prepare_data_OWID(country='United Kingdom', lockdown_date = None, URLnotfile = True):
    """
    extract covid.ourworldindata.org COVID-19 cases and deaths data by country from github urls
    
    otherwise similar to prepare_data() 
    
    
    ***but date labels all shifted by 1 day***
    
    
    """    

    #confirmed cases in time_series_covid19_confirmed_global.csv
    url_c = 'https://covid.ourworldindata.org/data/ecdc/total_cases.csv'
    file_c = 'C:/Users/Mark/Documents/Python/code/covid19/time_series_covid19_confirmed_global.csv'
    
    if URLnotfile:
        read_c = url_c #url_c or local file_c if saved already
    else:
        read_c = file_c #url_c or local file_c if saved already
        
    df_c = pd.read_csv(read_c)   #global confirmed cases
    df_c = df_c.T
    
    dates_Series = df_c.head(1).values
    dates_list = dates_Series.tolist()[0]
    
    df_c = df_c.tail(df_c.shape[0]-1)
    
    df_c.columns = dates_list
    
    df_c = df_c.reset_index(drop=False)

    df_c['Country/Region'] = df_c['index']

    df_cc = df_c.loc[(df_c['Country/Region']==country)]
    df_cc = df_cc.T; df_cc.columns=['cases']
    df_cc = df_cc.iloc[1:]
    df_cc = df_cc.iloc[:-1] 
    df_cc.index = pd.to_datetime(df_cc.index)
    
    #deaths in time_series_covid19_deaths_global.csv
    url_d = 'https://covid.ourworldindata.org/data/ecdc/total_deaths.csv'
    file_d = 'C:/Users/Mark/Documents/Python/code/covid19/time_series_covid19_deaths_global.csv'
    if (read_c==url_c):
        read_d = url_d #url_d or local file_d if saved already
    else:
        read_d = file_d

    df_d = pd.read_csv(read_d)   #global confirmed cases
    df_d = df_d.T
    
    dates_Series = df_d.head(1).values
    dates_list = dates_Series.tolist()[0]
    
    df_d = df_d.tail(df_d.shape[0]-1)
    
    df_d.columns = dates_list
    
    df_d = df_d.reset_index(drop=False)

    df_d['Country/Region'] = df_d['index']

    df_dc = df_d.loc[(df_d['Country/Region']==country)]
    df_dc = df_dc.T; df_dc.columns=['deaths']
    df_dc = df_dc.iloc[1:]
    df_dc = df_dc.iloc[:-1] 
    df_dc.index = pd.to_datetime(df_dc.index)
   
    # 2020-04-24 are some countries' new cases net of recoveries?
    # e.g. France has negative new cases    
    
    df = pd.concat([df_cc,df_dc], axis=1, sort=False)
    
    df['country'] = country
    df['latest_data_date'] = df_dc.index.max().strftime('%Y-%m-%d')
    
    if pd.isnull(lockdown_date):
        df = df[['country','latest_data_date','cases','deaths']]
    else:
        df['lockdown'] = lockdown_date    
        df = df[['country','latest_data_date','lockdown','cases','deaths']]
    
    df['new_deaths'] = df['deaths']-df['deaths'].shift(1)
    df['new_cases'] = df['cases']-df['cases'].shift(1)
    df.at[df.index[0],'new_deaths']=df.loc[df.index[0],'deaths']
    df.at[df.index[0],'new_cases']=df.loc[df.index[0],'cases']
    df['new_cases_rate'] = df['new_cases'] / (df['cases'].shift(1)+1e-10)
    df.at[df.index[0],'new_cases_rate']=0.0
 
    negative_rates_list = list(df.loc[df['new_cases_rate']<0].index)
    negative_rates_list = [d.strftime('%Y-%m-%d') for d in negative_rates_list]
    if negative_rates_list: #warn
        print(); print('--------------------------------------------------------------------')
        print('WARNING: check negative growth in new cases for dates',negative_rates_list)
        print('         -> new cases and growth rates for these dates have been set to zero')
        print(); print('--------------------------------------------------------------------')
        df.at[df['new_cases']<0,'new_cases']=0.0        
        df.at[df['new_cases_rate']<0,'new_cases_rate']=0.0   
        
    df = df[df.index>='22/01/2020'] #ensures same data range as Johns Hopkins data    
    
    return df

#---------------------------------------------------------------------------------

def fit_survival_negative_binomial(df, ew_halflife_days=50, verbose=True):
    """
    fits and adds parameters (s,n,p) each day from day 30 
    to the DataFrame df
    
    each day row in fit range [0,m] has weight  (1-alpha)**(m-row)
    where alpha = 1-exp(ln(0.5)/ew_halflife_days)
    
    returns df
    
    Notes
    =====
    bounds_tuple for (s, p, n) can be adjusted to ensure expected time to death,
    mu = n*p/(1-p), is within desired range. At present 2 < mu < 30 days
    
    """

    bounds_tuple = ((0.1,0.98),(0.0,0.1),(0.25,0.75),(6.,10.))   #(s,p,n) bounds in optimiser
    max_iterations = 50
    init_params_tuple = (0.75,0.001,0.4,10.0)
    
    #=================================================== fit each day
    
    start_loc = 30 #pointless fitting 3 parameters to less than 30 data points
    end_loc = df.shape[0]
    
    for i in range(start_loc, end_loc):
        fit_df = df.iloc[:i].copy()
        #weight to most recent experience using halflife parameter in days
        fit_df['weights'] = ew_halflife(n=i,halflife=ew_halflife_days) 
        res=fit_model(initparams=init_params_tuple, 
                      hyperparams=fit_df, 
                      bounds=bounds_tuple, 
                      maxiter=max_iterations)
        a,b,p,n=res[0]
        mean = nbinom.mean(n, p) #n*p/(1-p) #see scipy\stats\_discrete_distns.py line 210
        if verbose:
            print(df.index[i].strftime('%Y-%m-%d'),'parameters=',res[0],',',
                  round(mean,1),'days to death,', int(res[1]),'error')
        init_params_tuple = res[0]
        df.at[df.index[i],'a']=a
        df.at[df.index[i],'b']=b
        df.at[df.index[i],'p']=p
        df.at[df.index[i],'n']=n
    #===================================================

    return df


#---------------------------------------------------------------------------------
def fit_err(params, hyperparams):
    """
    error function used with fitting function fit_model()
    """
    a,b,p,n = params    #then s varies with time as a*(1-exp(bt)) 
    df = hyperparams
    
    if 'lockdown' in df.columns:   #may choose to only fit error after this date
        lockdown_date = pd.Timestamp(df['lockdown'].iloc[0]) 
        df = df.loc[df.index>=lockdown_date]   
        
    #update the model projection this set of parameters
    days_limit = df.index.shape[0]  #we limit this to prevent the fitter taking too long
    arr = np.array(range(days_limit))  #we limit analysis to days_limit after infection
    nbpr = lambda x: nbinom.pmf(x, n, p)
    negbin_probabilities = nbpr(arr)
    
    #pad the negbin_probabilities array with 0 in front
    #use np.pad(a, (1, 0), 'constant', constant_values=(0, 0))
    num_rows = df.index.shape[0]
    num_cols = negbin_probabilities.shape[0]+num_rows
    pmatrix = np.zeros((num_rows,num_cols))
    for m in range(num_rows):
        front_pad_cols = m
        back_pad_cols = num_rows-m
        pmatrix[m,:] = np.pad(negbin_probabilities, (front_pad_cols, back_pad_cols),
               'constant', constant_values=(0, 0))
        
    nmd_prob = np.matmul(df['new_cases'].to_numpy().reshape(1,-1).astype(float),
                         pmatrix.astype(float))
    s_array = a + b * np.arange(-1*num_rows+1,1)
    model_new_deaths = nmd_prob[:,:num_rows] * (1-s_array)
    new_deaths = df['new_deaths'].values.reshape(1,-1).astype(float)    
    error_squared = (new_deaths - model_new_deaths)**2 
    
    #penalise negative survival rate at start of fit
    error_squared += (1e6*min(s_array[0],0))**2 
    
    if 'weights' in df.columns:  #optional - exponentially weight errors
        weights = df['weights'].values.reshape(1,-1).astype(float)    
        error_squared = error_squared * weights 
    
    latest_error_squared = error_squared.sum()
    
    return latest_error_squared


#---------------------------------------------------------------------------------
def fit_model(initparams, hyperparams, bounds, maxiter):
    """
    optimiser associated with fit error function, fit_err()
    """
    from scipy import optimize as opt
    res  = opt.fmin_l_bfgs_b(fit_err, 
                             initparams, 
                             args=(hyperparams,),
                             approx_grad =True, 
                             bounds = bounds, 
                             maxiter = maxiter)
    #alternative optimiser with bounds on parameters:
    #res  = opt.fmin_slsqp(fit_err, initparams, args=(hyperparams,), bounds = bounds, iter=30)
    return res        

  
#---------------------------------------------------------------------------------
def project_new_cases(new_cases_df, halflife_days=5):
    """
    WLS projected new_cases_rate using halflife in days 
    The projection new_cases_rate(t) = exp(k + beta*t) is fitted via log transformation 
        
    returns (new_cases_df with fit added, k, beta)
    
    """

    import statsmodels.api as sm  #use weighted least squares to fit trend in new cases
    mask = (new_cases_df['new_cases'].cumsum()>100)
    X = np.arange(0,new_cases_df.loc[mask].shape[0])
    y = new_cases_df['new_cases_rate'].loc[mask].values.astype(float)
    y = np.log(1e-3 + y)  #ensure days with no new cases don't produce error or exert undue weight
    X = sm.add_constant(X)
    weights = ew_halflife(n=X.shape[0],halflife=halflife_days)
    wls_new_cases = sm.WLS(y, X, weights)
    results = wls_new_cases.fit()
    k = results.params[0]
    beta = results.params[1]
    new_cases_pred = X[:,0]*k + X[:,1]*beta
    new_cases_pred = np.exp(new_cases_pred)
    new_cases_df['new_cases_rate_fitted'] = np.nan
    new_cases_df.at[mask,'new_cases_rate_fitted'] = new_cases_pred.reshape((-1,1))
    new_cases_df['weights'] = np.nan
    new_cases_df.at[mask,'weights'] = weights
    
    return (new_cases_df, k, beta)

#-------------------------------------------------
def create_projection_df(params, df, project_new_cases_indicator=0):
    """
    adds to DataFrame df the negative binomial model deaths 
    (used to plot model projection for last set of parameters, params)
    
    extends df 100 days into the future and derives new deaths 
    based on parameters in params 
    (and new cases using parameters in df if project_new_cases_indicator is True)
    
    returns tuple of (completed DataFrame df, 
                      numpy array negbin_probabilities)
    """

    import warnings
    from pandas.core.common import SettingWithCopyWarning
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    from scipy.stats import nbinom

    a,b,p,n = params
    if 'lockdown' in df.columns:
        lockdown_date = df['lockdown'].iloc[0]
        df = df.loc[df.index>=lockdown_date]   #we only fit error after this date
        
    #add survival probabilities for dates where they apply
    mask = (df['new_cases'].cumsum()>100)    
    df.at[mask,'s'] = a + b * np.arange(-1*df.loc[mask].shape[0]+1,1)
    #hold survival rates constant in period before 100 cumulative cases
    df['s']=df['s'].fillna(method='bfill')

    #add another 100 days to index for projecting future cases and deaths
    next_dt = df.index[-1]+pd.DateOffset(1) 
    index_list = list(df.index)+list(pd.date_range(next_dt,periods=100, freq='D').values)
    index_list = [pd.Timestamp(d) for d in index_list]

    #reindex df to index_list to add 100 days projected deaths, etc.
    df = df.reindex(index_list)
    df['cases']=df['cases'].fillna(method='ffill')   #TODO add growth using cases_growth_rate
    df['deaths']=df['deaths'].fillna(method='ffill')
    #survival probabilities for future dates remain constant at current level
    df['s']=df['s'].fillna(method='ffill')
    df = df.fillna(0.0)
    
    days_limit = df.index.shape[0]  #we limit this to prevent the fitting taking too long
    arr = np.array(range(days_limit))  #we limit analysis to days_limit after infection
    nbpr = lambda x: nbinom.pmf(x, n, p)
    negbin_probabilities = nbpr(arr)
    
    if project_new_cases_indicator:   #add projection for new cases based on new_cases_rate
        k = df['k'].iloc[0]
        beta = df['beta'].iloc[0]*1
        mask = (df['new_cases'].cumsum()>100)
        X2 = np.arange(0,df.loc[mask].shape[0])
        X1 = np.ones(X2.shape[0])
        X = np.hstack((X1.reshape(-1,1),X2.reshape(-1,1)))
        new_cases_proj = X[:,0]*k + X[:,1]*beta
        df['new_cases_rate_projected'] = np.nan
        df.at[mask,'new_cases_rate_projected'] = np.exp(new_cases_proj.reshape((-1,1)))
        #apply these rates to last published accum cases and note under new_cases column
        latest_data_date = pd.Timestamp(df['latest_data_date'].iloc[0])
        mask = (df.index>latest_data_date)
        latest_cases = df.loc[~mask,'cases'].iloc[-1] 
        new_cases_Series = df.loc[mask,'new_cases']
        for ndx in new_cases_Series.index:
            new_cases_projected = latest_cases * df.loc[ndx,'new_cases_rate_projected']
            new_cases_Series.at[ndx] = new_cases_projected
            latest_cases += new_cases_projected 
        df.at[mask,'new_cases'] = new_cases_Series
        df.at[mask,'cases'] = df.loc[~mask,'cases'].iloc[-1]  + new_cases_Series.cumsum()
    else:  #new_cases remains zero so no deaths projected below for future new cases
        print('the negative binomial model projected for 100 days accounts for',
          round(100*negbin_probabilities[0:100].sum(),5),'% of future deaths')

    num_rows = days_limit
    num_cols = negbin_probabilities.shape[0]+num_rows
    pmatrix = np.zeros((num_rows,num_cols))
    for m in range(num_rows):
        front_pad_cols = m
        back_pad_cols = num_rows-m
        pmatrix[m,:] = np.pad(negbin_probabilities, (front_pad_cols, back_pad_cols), 
               'constant', constant_values=(0, 0))
        
    nmd_prob = np.matmul(df['new_cases'].to_numpy().reshape(1,-1).astype(float),
                         pmatrix.astype(float))
    
    s_array = df['s'].values
    model_new_deaths = nmd_prob[:,:num_rows] * (1-s_array)
    df['model_new_deaths'] = model_new_deaths.squeeze()
    if project_new_cases_indicator:   #add projection for new cases based on new_cases_rate
        mask = (df.index>latest_data_date)
        df.at[mask,'deaths'] = df.loc[~mask,'deaths'].iloc[-1] + df.loc[mask,'model_new_deaths'].cumsum()
             
    return df, negbin_probabilities

#---------------------------------------------------------------------------------
def find_median_halflife_days(new_cases_df, HLD_list = [2,3,4,5,6,7,8,9,10]):
    """
    fit new_cases_rate(t) = exp(k + beta * t) using weighted least squares 
    and choose the median beta from halflife_days weighting in HLD_list
    record fit parameters in fit_dict, indexed on HLD

    2020-06-15: see docs. We now fit using 7-day rolling average of new_cases_rate. 
    
    returns median_halflife_days, fit_dict

    """
    fit_dict={}  #to store fitted parameters, indexed on HLD

    new_cases_df_MA = new_cases_df.copy() #first make a deep copy
    #reduce effect of weekday reporting pattern by using 7-day rolling mean new_cases_rate
    new_cases_df_MA['new_cases_rate'] = new_cases_df_MA['new_cases_rate'].rolling(window=7).mean()
    new_cases_df_MA['new_cases_rate'] = new_cases_df_MA['new_cases_rate'].fillna(0.0)

    for HLD in HLD_list:
        new_cases_df_MA,k,beta = project_new_cases(new_cases_df_MA, halflife_days = HLD)
        fit_dict[HLD]={'k': k, 'beta': beta}
    beta_list = [fit_dict[x]['beta'] for x in HLD_list]
    median_beta = np.median(beta_list)
    median_HLD = [k for k in fit_dict.keys() if abs(fit_dict[k]['beta']-median_beta)<1e-5]
    median_HLD = median_HLD[0]
    median_halflife_days = median_HLD

    return median_halflife_days, fit_dict

#---------------------------------------------------------------------------------
def compare_new_cases_rate_beta(country_list, last_n_days=10):
    """

    returns dict of {country: DataFrame indexed by date of 
                              fitted model for new_cases_rate for last_n_days}
    
    NB: assumes we have already saved latest data to local today using save_data()

    """
    summary_dict = {}
    for country in country_list:
        df = prepare_data(country = country, lockdown_date = None, URLnotfile = False)
        country_df = pd.DataFrame()
        for j in range(last_n_days):
            new_cases_df = df[['new_cases','new_cases_rate']].head(df.shape[0]-j)
            mask = (new_cases_df['new_cases'].cumsum()>=100) #only fit after 100 cases
            new_cases_df = new_cases_df.loc[mask]
            median_HLD, fit_dict = find_median_halflife_days(new_cases_df, HLD_list = [2,3,4,5,6,7,8,9,10])    
            summary_date = new_cases_df.index[-1]            
            country_df.at[summary_date,'median_HLD'] = median_HLD
            country_df.at[summary_date,'k'] = fit_dict[median_HLD]['k']
            country_df.at[summary_date,'beta'] = fit_dict[median_HLD]['beta']
        summary_dict[country] = country_df
    
    print(list(summary_dict.keys()),'added to summary_dict')
    
    return summary_dict


#---------------------------------------------------------------------------------
def compare_new_cases_rate_beta_test(country_list, last_n_days=10):
    """
    test scenario: 
    how negative would fitted beta be had new cases stayed constant in absolute terms?
    RESULT: at 2020-04-28 the fitted beta would be around -2.5%
            i.e. higher than we are seeing in data
    """
    summary_dict = {}
    for country in country_list:
        df = prepare_data(country = country, lockdown_date = None, URLnotfile = False)
        country_df = pd.DataFrame()
        for j in range(last_n_days):
            new_cases_df = df[['deaths','cases','new_cases','new_cases_rate']].head(df.shape[0]-j)
            mask = (new_cases_df['new_cases'].cumsum()>=100) #only fit after 100 cases
            denominator = new_cases_df.loc[~mask,'new_cases'].sum()
            new_cases_df = new_cases_df.loc[mask]
            new_cases_start = new_cases_df.head(3)['new_cases'].mean()
            new_cases_df['new_cases'] = new_cases_start 
            new_cases_df['cases'] = denominator +new_cases_df['new_cases'].cumsum()
            #test effect of deaths remaining constant at this level
            new_cases_df['new_cases_rate'] = new_cases_df['new_cases'] / (new_cases_df['cases'].shift(1)+1e-10)
            new_cases_df['new_cases_rate'] = new_cases_df['new_cases_rate'].fillna(method='bfill')
            median_HLD, fit_dict = find_median_halflife_days(new_cases_df, HLD_list = [2,3,4,5,6,7,8,9,10])    
            summary_date = new_cases_df.index[-1]            
            country_df.at[summary_date,'median_HLD'] = median_HLD
            country_df.at[summary_date,'k'] = fit_dict[median_HLD]['k']
            country_df.at[summary_date,'beta'] = fit_dict[median_HLD]['beta']
        summary_dict[country] = country_df
    
    print(list(summary_dict.keys()),'added to summary_dict')
    
    return summary_dict


#---------------------------------------------------------------------------------
def produce_seasonality_dict(proj_df):
    """
    returns seasonality_dict, a dict of weekday seasonality multipliers for
    daily model_new_deaths in proj_df

    """    
    
    #identify when cumulative deaths exceed 100 for fitting purposes    
    mask = (proj_df['deaths']>100) & (proj_df['new_deaths']>0)
    seasonality_df = proj_df.loc[mask]
    seasonality_df['Date'] = seasonality_df.index
    seasonality_df['multiplier'] = proj_df['new_deaths']/proj_df['model_new_deaths']
    seasonality_df['log_multiplier'] = np.log(seasonality_df['multiplier']+0.001)
    #Monday is weekday 0
    seasonality_df['weekday'] = seasonality_df['Date'].apply(lambda x: x.weekday())
    seasonality_dict={}
    grouper = seasonality_df.groupby('weekday')
    for weekday, wdf in grouper:
        seasonality_dict[weekday] = np.dot(wdf['log_multiplier'], wdf['weight'])/(wdf['weight'].sum())
    #normalise the seasonality dict
    mu = np.array(list(seasonality_dict.values())).mean()
    seasonality_dict = {k: (seasonality_dict[k] - mu) for k in seasonality_dict.keys()}
    #transform seasonality dict back to multipliers
    seasonality_dict = {k: np.exp(seasonality_dict[k]) for k in seasonality_dict.keys()}
    return seasonality_dict


#---------------------------------------------------------------------------------
def investigate_seasonality(selected_country,image_path='C:/Users/Mark/Documents/Python/code/covid19/latest/'):
    """
    calculates and plots intra-week seasonality multipliers for daily deaths
    for selected_country, saving the results to image_path
    """    
    
    df = prepare_data(country = selected_country, lockdown_date = None, URLnotfile = False)
  
    ew_halflife_days=20
    df['weight'] = ew_halflife(df.shape[0],ew_halflife_days)
    df = fit_survival_negative_binomial(df.copy(), ew_halflife_days=ew_halflife_days, verbose=True)
    a,b,p,n = tuple(df[['a','b','p','n']].iloc[-1])   #parameters fitted to latest date row
    params = (a,b,p,n)
    proj_df,negbin_probabilities = create_projection_df(params=params, df=df.copy(), 
                                                        project_new_cases_indicator=False)
    
    proj_df['weight'] = ew_halflife(proj_df.shape[0],ew_halflife_days)
    seasonality_dict = produce_seasonality_dict(proj_df)
    seasonality_Series = pd.Series(list(seasonality_dict.values()),['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    ax=seasonality_Series.plot(title='daily seasonality, '+selected_country,figsize=(10,6))
    ax.spines['bottom'].set_position(('data', 1.0))
    ax.patch.set_edgecolor('grey')  
    ax.patch.set_linewidth('1')
    plt.savefig(image_path+selected_country.upper()+'_daily_seasonality.png')
    plt.show()
    return seasonality_dict
    
#---------------------------------------------------------------------------------
def analyse_country(selected_country,image_path='C:/Users/Mark/Documents/Python/code/covid19/latest/'):
    """
    plots the following for string selected_country...

        1) cases, deaths to date;
        2) latest fitted negative binomial parameters;
        3) evolution of fitted survival rates in last 30 days;
        4) new_cases_rate and median fit;
        5) show the mid projection, cases and deaths; 
        6) show the mid deaths only projection with 90% confidence bounds

    ...saving the results to image_path
    """    
    
    df = prepare_data(country = selected_country, lockdown_date = None, URLnotfile = False)
  
    print(df.tail())
    
    ew_halflife_days=20
    print('ew_halflife_days =',ew_halflife_days)
    print()
        
    df = fit_survival_negative_binomial(df.copy(), ew_halflife_days=ew_halflife_days, verbose=True)
    a,b,p,n = tuple(df[['a','b','p','n']].iloc[-1])   #parameters fitted to latest date row

    params = (a,b,p,n)

    proj_df,negbin_probabilities = create_projection_df(params=params, df=df.copy(), 
                                                        project_new_cases_indicator=False)

    proj_df['weight'] = ew_halflife(proj_df.shape[0],ew_halflife_days)
    seasonality_dict = produce_seasonality_dict(proj_df)
    
    proj_df['new_deaths'] = proj_df['new_deaths'].replace(0.0, np.nan) #don't plot zero values
    proj_df['new_cases'] = proj_df['new_cases'].replace(0.0, np.nan) #don't plot zero values
    
    
    
    #======================                          plot cases, deaths to date
    latest_data_date_str = df['latest_data_date'].iloc[0]
    ax = df[['new_deaths']].iloc[40:].plot(title=selected_country
           +' published daily new cases and deaths as at '
           +latest_data_date_str, figsize=(10,6))
    plt.ylabel('new deaths')
    plt.ylim(ymin=0)
    ax2 = df['new_cases'].iloc[40:].plot(secondary_y=True, ax=ax, 
            color='black', linestyle='dotted',label='new_cases')
    plt.ylabel('new cases')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc = 'upper left')  #'lower right'    
    plt.ylim(ymin=0)
    plt.show()
    #======================


    #======================            latest fitted negative binomial parameters
    mean = round(n*(1-p)/p,1)
    print(selected_country,'mean time until death', str(round(mean,1))
             ,' days between positive test result and death')
    ax=pd.Series(negbin_probabilities[0:20]).plot.bar(figsize=(6,3.75))
    
    ax.set_title(selected_country+' negative binomial probabilities for model fit at '
             +latest_data_date_str,fontsize=9.5)
    plt.show()
    #======================


    #====================== evolution of fitted survival rates, s, last 30 days
    survivalrate_Series = a + b * np.arange(-1*df.shape[0]+1,1)
    survivalrate_Series = survivalrate_Series*100 #in percent
   
    sdf = pd.DataFrame(survivalrate_Series, columns=['s'], index=df.index)
    ax = sdf[['s']].tail(30).plot(
            title='fitted survival rate, '+selected_country,
            ylim=(50,100), 
            figsize=(8,5.5))
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.grid()
    plt.savefig(image_path+selected_country.upper()+'_survival.png')
    plt.show()
    #======================


    # we fit new_cases_rate(t) = exp(k + beta * t) using weighted least squares 
    # and choose the median beta from halflife_days weighting in HLD_list
    new_cases_df = df[['new_cases','new_cases_rate']]
    mask = (new_cases_df['new_cases'].cumsum()>=100)
    new_cases_df = new_cases_df.loc[mask]
    median_HLD, fit_dict = find_median_halflife_days(new_cases_df, HLD_list = [2,3,4,5,6,7,8,9,10])
    
    median_beta = fit_dict[median_HLD]['beta']
    
    print('use median beta in projection of',round(median_beta,4),'for halflife_days of',median_HLD)
    print()

    new_cases_df = df[['new_cases','new_cases_rate']].dropna().copy()
    new_cases_df,k,beta = project_new_cases(new_cases_df, halflife_days = median_HLD)

    print()
    print(selected_country+' new cases growth rate and fitted exponential curve')
    print()
    #====================== plot new_cases_rate and median fit
    mask1 = (new_cases_df['new_cases'].cumsum()>=100)
    mask2 = (new_cases_df.index>new_cases_df['new_cases'].index[-21]) #only last 20 observations

    new_cases_df['new_cases_rate'] = new_cases_df['new_cases_rate'].rolling(window=7).mean()
    new_cases_df['new_cases_rate'] = new_cases_df['new_cases_rate'].fillna(0.0) 

    fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))
    new_cases_df[['new_cases_rate','new_cases_rate_fitted']].loc[mask1].plot(ax=ax)
    ax.set_title('new cases and fitted model exp('+str(round(k,4))+'+'+str(round(beta,5))+'t)', fontsize=11)
    new_cases_df[['new_cases_rate','new_cases_rate_fitted']].loc[mask2].plot(ax=ax2)
    #ax2.set_title('new_cases_rate_fitted(t) = exp(k+t.beta)', fontsize=11) 
    ax2.set_title('latest 20 observations', fontsize=11)
    fig.tight_layout()
    plt.show()
    #======================

        
    df['k'] = k
    df['beta'] = beta
    proj_df,negbin_probabilities = create_projection_df(params=(a,b,p,n), df=df, 
                                                        project_new_cases_indicator=True)
        
    #======================  show the mid projection
    plot_df = proj_df[['model_new_deaths']]
    latest_data_date = pd.Timestamp(df['latest_data_date'].iloc[0])
    mask = (proj_df.index>latest_data_date)
    plot_df.at[mask,'model_new_cases']= proj_df.loc[mask,'new_cases']
    plot_df.at[~mask,'new_deaths'] = proj_df.loc[~mask,'new_deaths']
    plot_df.at[~mask,'new_cases'] = proj_df.loc[~mask,'new_cases']

    if (median_beta>-0.01):   #only project 30 days if cases growth rate not under control
        plot_df=plot_df.head(plot_df.shape[0]-70)

    title_str = selected_country+' model deaths (with projected new cases)'
    ax = plot_df[['new_deaths','model_new_deaths']].iloc[40:].plot(title=title_str, figsize=(11.7,7))
    plt.ylabel('daily deaths')
    plt.ylim(ymin=0)        
    ax2 = plot_df[['new_cases','model_new_cases']].iloc[40:].plot(secondary_y=True, ax=ax, 
                 color=['black','grey'], linestyle='dotted',label='new_cases')
    plt.ylabel('new cases')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc = 'best')  #'lower right'
    plt.ylim(ymin=0)        
    plt.savefig(image_path+selected_country.upper()+'_cases_deaths'+'.png')
    plt.show()
    #======================

    print()
    current_deaths = int(df['deaths'].iloc[-1])
    accum_deaths = int(proj_df['deaths'].iloc[-1])

    title_text='cumulative deaths by '+proj_df.index[-1].strftime('%Y-%m-%d')+' of '+str(accum_deaths)
    title_text+=', deaths to date '+ str(current_deaths)+' = ' +str(int(100*current_deaths/accum_deaths))+'%'
    print(title_text)
    print()

    #weekday seasonality for 'model_new_deaths'
    proj_df['Date'] = proj_df.index
    proj_df['weekday'] = proj_df['Date'].apply(lambda x: x.weekday())
    proj_df['seas_multiplier'] = proj_df['weekday'].apply(lambda x:seasonality_dict[x])
    proj_df['model_new_deaths'] *=  proj_df['seas_multiplier'] 
    

    if (median_beta<-0.01):
        #======================  show confidence bounds only if case growth rate<0
        #90% confidence bounds assuming range between 5th and 95th percentile of residuals
        if (selected_country == "Sweden") or (selected_country == "Japan"):
            threshold_daily_deaths = 20
        else:
            threshold_daily_deaths = 100
        latest_data_date = proj_df['latest_data_date'].iloc[0]
        mask = (proj_df['model_new_deaths']>threshold_daily_deaths) & (proj_df.index<=latest_data_date)
        proj_df.at[mask,'error'] = proj_df.loc[mask,'new_deaths']-proj_df.loc[mask,'model_new_deaths']
        proj_df.at[mask,'error'] = proj_df.loc[mask,'error'] / proj_df.loc[mask,'model_new_deaths']
        l_bound, u_bound = np.percentile(proj_df.loc[mask,'error'].values, 
                                                 [5,95], interpolation = 'linear')
        #rebalance bounds around zero
        upper_bound = (u_bound- l_bound)/2.
        lower_bound = (l_bound- u_bound)/2.    
    
        plot_df = proj_df[['model_new_deaths']]
        latest_data_date = pd.Timestamp(df['latest_data_date'].iloc[0])
        mask = (proj_df.index>latest_data_date)  
        plot_df.at[mask,'5% bound new_deaths'] = proj_df.loc[mask,'model_new_deaths']*(1+lower_bound)
        plot_df.at[mask,'95% bound new_deaths'] = proj_df.loc[mask,'model_new_deaths']*(1+upper_bound)
        plot_df.at[mask,'model_new_cases']= proj_df.loc[mask,'new_cases']
        plot_df.at[~mask,'new_deaths'] = proj_df.loc[~mask,'new_deaths']
        plot_df.at[~mask,'new_cases'] = proj_df.loc[~mask,'new_cases']
        
        title_str = selected_country+' model deaths with 90% confidence limits and daily seasonality,'+'\n '+title_text
        ax = plot_df[['new_deaths','model_new_deaths']].iloc[40:].plot(title=title_str, figsize=(11.7,7))
        ax.fill_between(plot_df['5% bound new_deaths'].index, plot_df['5% bound new_deaths'], plot_df['95% bound new_deaths'], 
                        color='orange', alpha=.2)   
    
        #indicate diminishing confidence in the confidence intervals
        for x in np.arange(-5,-95,-2):
            ax.fill_between(plot_df['5% bound new_deaths'].index[x:], plot_df['5% bound new_deaths'].tail(x*-1), plot_df['95% bound new_deaths'].tail(x*-1), 
                        color='white', alpha=.1)
        
        plt.ylabel('daily deaths')
        plt.ylim(ymin=0)        
        plt.savefig(image_path+selected_country.upper()+'.png')
        plt.show()
        #======================
        return ('analyse_country() for '+str(selected_country)+' completed')
    
#---------------------------------------------------------------------------------    
def investigate_seasonality_mp(country_list):
    pool = multiprocessing.Pool()
    results = pool.map(investigate_seasonality, country_list)
    pool.close()
    pool.join()
    for result in results:
        print(result)

#---------------------------------------------------------------------------------    
def analyse_country_mp(country_list):
    pool = multiprocessing.Pool()
    results = pool.map(analyse_country, country_list)
    pool.close()
    pool.join()
    for result in results:
        print(result)

#________________________________________________________________________________
if __name__ == "__main__":
    
    country_list = ['United Kingdom','Italy','Spain','US','Sweden','Brazil','Germany','France', 'South Africa','Japan']
    
    original_DPI = plt.rcParams["figure.dpi"]
    plt.rcParams["figure.dpi"] = 100  #higher DPI plots

    save_data()  #extract Johns Hopkins data and save locally

    image_path = 'latest/'
    image_path = 'C:/Users/Mark/Documents/Python/code/covid19/' +image_path

    '''   OLD: ***serial processing getting slow so code commented out ***
    for selected_country in country_list:
        analyse_country(selected_country,image_path)
        investigate_seasonality(selected_country,image_path) #saves charts
    '''    

    print('using multiprocessing module to analyse each country - be patient!')
    analyse_country_mp(country_list)

    '''   OLD: ***serial processing getting slow so code commented out ***
    for selected_country in country_list:
        investigate_seasonality(selected_country,image_path) #saves charts
    '''    
    investigate_seasonality_mp(country_list)
    

    #====================== plot evolution of beta parameters across countries
    country_list = ['United Kingdom','Italy','Spain','US','Sweden','Brazil']

    summary_dict = compare_new_cases_rate_beta(country_list=country_list, last_n_days=40)
    beta_df = pd.DataFrame()
    for country in country_list:
        cSeries = summary_dict[country]['beta']; cSeries.name=country
        beta_df = pd.concat([beta_df,cSeries], axis=1)
    beta_df = beta_df.sort_index()     
    ax = beta_df.plot(figsize=(10.5,6.25),ylim=(-0.1,0.0), title = 
                 'beta parameter for new cases rate curves exp(k+beta.t) fitted up to each Date on x-axis')
    plt.savefig(image_path+'compare_beta_new_cases_growth.png')
    plt.show()
    #====================== 
    
        
    '''
    #test extent fitted beta parameter would have been negative had new cases stayed constant in absolute terms
    country_list = ['United Kingdom','Italy','Spain','US','Sweden','Brazil'] #, 'Germany']
    summary_dict2 = compare_new_cases_rate_beta_test(country_list=country_list, last_n_days=20)
    beta_df = pd.DataFrame()
    for country in country_list:
        cSeries = summary_dict2[country]['beta']; cSeries.name=country
        beta_df = pd.concat([beta_df,cSeries], axis=1)
    beta_df = beta_df.sort_index()     
    beta_df.plot(figsize=(10.5,6.25),ylim=(-0.1,0.0), title = 
                 'TEST beta parameter had new cases stayed constant in absolute terms')
    '''
    


    #restore orginal plot density parameters
    plt.rcParams["figure.dpi"] = original_DPI 
    
    print('data available for:')
    print(investigate_data())