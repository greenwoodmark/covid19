# COVID19
Negative binomial model for projecting COVID-19 cases and deaths data by country, provided by Johns Hopkins University CSSE at [CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19).

The function `main()` fits the model to the latest date and charts the results. These charts are saved daily in the [latest](latest/) and archived to the [archive](archive/) directory. The generous 1MB limit on a github repo accommodates about a year of historical charts.

The model estimates the new deaths at each future date $t$ based on new cases declared on each previous date:

$$
\operatorname{E} \bigl[  {nd}_t  \bigr] = \sum_{i=1}^t {nc}_i  \left( 1-s \right)  Pr\bigl[ \text{ dies  at }  t \text{ } \mid \text{ } {nc}_i \bigr ]
$$

where:

&emsp; ${s}$ = probability of survival for a new case 

&emsp; ${nd}_t$ = new deaths on date $t$

&emsp; ${nc}_t$ = new cases on date $t$

The negative binomial distribution is used to describe this conditional probability. We assume that the lag between a positive test result (i.e. creating a new case) and death due to COVID-19 follows a negative binomial distribution with parameters $n$ and $p$. This can be interpreted as the probability there will be $k$ failures until the $n$-th success for $k+n$ independent and identically distributed trials, each with probability of success $p$. In the above expression $k$ equals the lag, $t-i$, so that:

$$
Pr \bigl[ \text{ dies  at }  t \text{ } \mid \text{ } {nc}_i \bigr ]  =   {{t-i+n-1} \choose { n-1 }} p^{n} (1-p)^{t-i} 
$$



For the evolution of the model, see in [docs](docs/)  `20200429 Squashing the sombrero - negative binomial model for COVID-19 deaths.pdf` and the addition of time-varying survival rates as documented in `20200512 COVID-19 deaths projection now uses time-varying survival rate.pdf`. 
In order to address weekday patterns for deaths and cases, changes were made as detailed in `20200519 weekday seasonality added to COVID-19 deaths projection.pdf`, `20200615 allowing for new cases seasonality in the COVID-19 deaths projection.pdf` and `20200630 allowance for improvement in survival rates in COVID-19 deaths projection.pdf`   
