# COVID19 negative binomial model
Negative binomial model for projecting COVID-19 cases and deaths data by country, provided by Johns Hopkins University CSSE at [CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19).

## Published daily new cases and deaths
For example, the new cases and deaths experience to date for Australia is:
![UK_cases_deaths](https://github.com/greenwoodmark/covid19/blob/master/latest/AUSTRALIA_cases_deaths.png)

## Time-varying survival rates
The model fits a time-varying survival rate and a negative binomial model for days until death given a new case does not survive:
![UK_survival](https://github.com/greenwoodmark/covid19/blob/master/latest/AUSTRALIA_survival.png)
![UK_probabilities](https://github.com/greenwoodmark/covid19/blob/master/latest/AUSTRALIA_probabilities.png)

## Projected new cases and deaths
The fitted projection for the model is then (only shown if new cases growth rate is under control): 
![UK](https://github.com/greenwoodmark/covid19/blob/master/latest/AUSTRALIA.png)

## Weekday reporting patterns
This projection shows a characteristic weekday reporting pattern:
![UK_daily_seasonality](https://github.com/greenwoodmark/covid19/blob/master/latest/AUSTRALIA_daily_seasonality.png)

## Trends in new cases growth rates
The new cases growth rate is critical. The chart below compares the evolution of this rate across countries and against the R0=1 reproduction rate boundary:
![compare_beta_new_cases_growth](https://github.com/greenwoodmark/covid19/blob/master/latest/compare_beta_new_cases_growth.png)
