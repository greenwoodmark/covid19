# COVID19
Negative binomial model for projecting COVID-19 cases and deaths data by country, provided by JHU CSSE at https://github.com/CSSEGISandData/COVID-19.

The code in `main()` fits the model to the latest date and charts the results. These charts are saved daily in the [latest](latest/) and archived to the [archive](archive/) directory. The generous 1MB repo limit on github can accommodate about a year of historical charts.


For the evolution of the model, see in docs directory `20200429 Squashing the sombrero - negative binomial model for COVID-19 deaths.pdf` and the addition of time-varying survival rates as documented in `20200512 COVID-19 deaths projection now uses time-varying survival rate.pdf`. 
In order to address weekday patterns for deaths and cases, changes were made as detailed in `20200519 weekday seasonality added to COVID-19 deaths projection.pdf`, `20200615 allowing for new cases seasonality in the COVID-19 deaths projection.pdf` and `20200630 allowance for improvement in survival rates in COVID-19 deaths projection.pdf`   
