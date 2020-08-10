# COVID19 negative binomial model
Negative binomial model for projecting COVID-19 cases and deaths data by country, provided by Johns Hopkins University CSSE at [CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19).

## Published daily new cases and deaths
For example, the new cases and deaths experience to date for the USA is:
![UK_cases_deaths](latest/UK_cases_deaths.png)

## Time-varying survival rates
The model fits a time-varying survival rate and a negative binomial model for days until death given a new case does not survive:
![UK_survival](latest/UK_survival.png)
![UK_probabilities](latest/UK_probabilities.png)

## Projected new cases and deaths
The fitted projection for the model is then: 
![UK](latest/UK.png)

## Weekday reporting patterns
This projection shows a characteristic weekday reporting pattern:
![UK_daily_seasonality](latest/UK_daily_seasonality.png)

## Trends in new cases growth rates
The new cases growth rate is critical. The chart below compares the evolution of this rate across countries and against the R0=1 reproduction rate boundary:
![compare_beta_new_cases_growth](latest/compare_beta_new_cases_growth.png)