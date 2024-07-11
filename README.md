# MIE1624_Final_Project
 
Working with the COVID-19 Vaccination Dataset

Steps:

Import all of the libraries

Convert the csv file to a pandas dataframe and convert the first three columns to a numeric column so Python can process it.

**Part 1: Data Cleaning**:

First, we see how many NAN values there are in the dataset.

Drop the rows with NAN values in total vaccinations since it doesn't make sense to include them

Plot the COrrelation Map

Fill NANs with 0 to allow us to check p-values for Mann-Whitney and t-tests

Delete the first three columns before MW and t-tests: Location, area code, and date

Ensure the alternate variable in Mann-Whitney U is 2-sided, is p is less than 0.05, reject the hypothesis, and fill the missing value with 0

Run stat distribution, and t-test, if any values are greater than 0.05

Created a double for loop to run the stats.mannwhitneyu(column1[i].values.reshape)-1,1), column2[i].values.reshape)-1,1), plotted on a heat map

![image](https://github.com/user-attachments/assets/f3088762-3299-4e0f-ab98-23edfd000756)

 As can be seen from our correlation data,  the values of the total_vaccinations column are mostly the same as the people_vaccinated column. Since p is essentially 0, we can see that the columns are related to one another

Will perform my method to fill in missing NAN values:

It looks like total_vaccinated = people_vaccinated + people_fully_vaccinated + total_boosters (based off Canada data)

Repeat the same procedures per hundred, there should be the same trend but different ratios based on country population size

Need to replace data with corresponding values, values are less than 0.05 in the Mann-Whitney U test, fill values with 0

**Part 2: Data Visualization and Exploratory Data Analysis**

Vaccines for COVID-19 have been in use since December 2020, and have been implemented in more and more countries over time.

We can observe the **locations with the most people vaccinated,** however, this doesn't take into account the country's population.

Grouping the data by country and then sorting the data by the max value from highest to lowest:

![image](https://github.com/user-attachments/assets/9cd10cc0-bfc0-462c-815f-d64bcd04ee54)

From this, we can see that China and India have the highest number of people vaccinated. This makes sense since those two countries have the largest population. This doesn't necessarily mean it has the highest vaccination rate (the percentage of people vaccinated / total population). The ratio of those vaccinated to those that are not matters when considering the rate of transmission, and emergence of new variants.

 We can also observe that remote island countries such as Pitcairn have the lowest amount of vaccinated people, this may be due to the remote location of the island and thus lack of resources to safely transfer the vaccine there. 

**Which countries have the best and most effective vaccination rollout?**

We can observe which countries have the most number of people vaccinated in a single day. This tells us whether vaccines are available in the country and can tell us whether the vaccines are enforced or encouraged by the people in power.

![image](https://github.com/user-attachments/assets/92ea4d57-cd0f-4971-9771-e49ae943046e)

We can see that the countries that have a bigger population, more infrastructure and resources, and bigger vaccine supply, appear to be becoming more vaccinated as the days go by. It seems like GDP is not a factor here, as many of the majority of the countries are either developing or third-world countries such as India and Brazil. 

**Plotting this data to see the progression in daily vaccinations over time:**

![image](https://github.com/user-attachments/assets/7412b13f-f24a-4725-9b16-3911a7c592ae)

We can see that China and India are still increasing their vaccination rate exponentially, whilst the US and Mexico are vaccinating their populations slowly but steadily. This is probably due to the enforcement of mandatory vaccines as well as the availability of vaccines now.

We can also see that the lines for Pakistan and Bangladesh end much earlier than any other country. It seems that the graph stops whenever the daily vaccinations start to decrease, so the graph shows whether the maximum daily vaccination has been reached and if the vaccine rollout has slowed down.

Perhaps their vaccine supplies run out or perhaps those in power have stopped enforcing the vaccines. We can see that for most countries, the rate of vaccination (the slope of the line) is still going up.  

**Plotting Vaccination Rate**

![image](https://github.com/user-attachments/assets/e6138b19-4382-46b9-8f7d-012de2046ca3)

Gibraltar is not looked at since it does not make sense to have more than a 100% vaccination rate. 

It seems that small countries such as Pitcairn and wealthy countries such as UAE and Singapore have very high vaccination rates.

**Part 3: Model Selection and Fitting to the Data**

ARIMA Model will be used to help predict the COVID-19 Vaccination rates in the next 50 days by allowing us to easily analyze the time series of the current vaccination rates. This is done for Canada and the US.

We want the time series stationary to make it not depend on the time at which the series is analyzed. To make a non-stationary time series stationary, we use differencing operations for stabilizing the mean of the series where we calculate the differences in a back-to-back manner. This helps in removing (or minimizing) trends and seasonality.

Sometimes first-order differencing of time series may not remove all the trend and seasonality and it is required to difference the data one more time (second-order differencing of time series) to get a stationary time series.

After analyzing the Autocorrelation plot, a second-order differencing is applied to make the time series stationary.

![image](https://github.com/user-attachments/assets/10465aa6-c5f5-4857-a83c-c1227334625e)

![image](https://github.com/user-attachments/assets/032ba2c6-a226-488a-8f21-957692339720)

From the second-order differencing autocorrelation, we can see that most of the points fit within the light-shaded region which means that our model is stationary. 

An ARIMA model without seasonality is represented as ARIMA(p,d,q), p is the count of AR terms, d is the count of differences needed (in this example, we see that d=2 is needed)to make the time series as a stationary one and q is the count of MA terms. Then we find the best ARIMA model to fit. Akaike’s Information Criterion (AIC), is observed here to identify the order of the best ARIMA model. The lowest AIC value represents the best model.

To find the optimal value of differencing (we assumed d=2 previously by plotting the time series), we can use the Augmented Dickey-Fuller (ADF) test.

Applying autoARIMA to find the best parameters to minimize AIC instead of doing it manually:
Best model for Canada:  ARIMA(5, 2, 2)
Best model for US:  ARIMA(5, 2, 4)

Canada Residual Plot:

![image](https://github.com/user-attachments/assets/4dd7b421-733c-45ff-814f-7a38640a7a4b)

US Residual Plot:

![image](https://github.com/user-attachments/assets/d975e58a-7e5a-45e9-8e9d-b477a5119e6c)

Vaccination Forecast for Canada Code:

![image](https://github.com/user-attachments/assets/97b24333-d6bc-4fe2-8539-041c0cdb4b77)

![image](https://github.com/user-attachments/assets/06ef5c7e-63a9-4760-998c-431c68c11e48)

![image](https://github.com/user-attachments/assets/631bfc85-c6b4-4d21-a2d0-82b3f277c4b0)

From our base case model, we can predict that 30,770,417 people will be vaccinated in Canada by the end of the next 50 days (from the last day of the dataset). This means 618,211 more people will be vaccinated.

Doing the same for the US:

![image](https://github.com/user-attachments/assets/1ed30fb3-413e-4f3b-a039-5a06c4549121)

![image](https://github.com/user-attachments/assets/b9fc2a00-9b07-4218-9afc-8ebca6ab720c)

From our base case model, we can predict that 240,826,270 people will be vaccinated in the US by the end of the next 50 days (from the last day of the dataset). This means 10,527,526 more people will be vaccinated.

There may be many factors or features affecting the vaccination rate in a country. This simple time series analysis by the ARIMA forecasting equation only considers a linear relationship between the days and vaccination rate in which the predictors comprised of lags of the dependent feature (here vaccination rate) and lags of the forecasting errors.

**Part 4 - Relating COVID-19 Vaccination Rates to a Second Dataset - Canada**

This dataset shows the number of ICU patients and people who have died or recovered from COVID

![image](https://github.com/user-attachments/assets/e86640c9-1c38-429e-a06f-9cc72368f7dc)

Plotting the number of deaths vs the amount of people vaccinated:

![image](https://github.com/user-attachments/assets/89e51aa1-ab7e-42d4-a400-38ee12f8c724)

From this graph, we can see that before the vaccine came out, there were many deaths related to COVID-19. This is because it took a while for the vaccine rollout to occur. As vaccines progressively came out, the rate of deaths decreased, and we can see the slope of deaths decreasing. Lastly, the slope of deaths is exponentially increasing as of now, this may be because new variants are out along with less strict lockdown regulations causing the vaccine to lose its effectiveness. 

![image](https://github.com/user-attachments/assets/63b8be30-b373-44f3-a91c-a3303e929181)

From this graph, we can see that before the vaccine came out, there were many people placed into icu treatments. Even as the vaccines began to roll out,
the number of icu patients increased until a large majority of Canadians got the vaccines. We see the number of ICU patients decrease when most Canadians are vaccinated but now it's starting to increase. This may be due to new variants along with less strict lockdown regulations causing the vaccine to lose its effectiveness as well.

Plotting Normalized Values against time:

![image](https://github.com/user-attachments/assets/b8230d92-d14e-4bee-8638-8d7ae4313afe)

We can see that the rate of people getting vaccinated in Canada is slowing down, the amount of deaths is somewhat linear, and the number of icu patients is decreasing.  

After running a t-test to show the relationship between the number of people vaccinated and the total deaths and icu patients is related: Since P-values are essentially 0, this means that the features are statistically significant (the differences between your groups are significant).

Will use vaccination rate to predict the amount of deaths and icu patients in the next 50 days:

Estimate the number of deaths and icu patients using linear regression.

![image](https://github.com/user-attachments/assets/3fceb113-5387-4902-b925-a3044fda07e9)

This model doesn't make sense since we see the forecasted number of deaths lower than the last recorded amount of deaths. This model can be fine-tuned by using a different regression model. We can also add such that the first value should coincide with the last recorded amount of deaths.

![image](https://github.com/user-attachments/assets/70508305-1150-4531-bc9a-200ea3f6b483)

We can see that the forecasted amount of icu patients is higher. This model can be fine-tuned by using a different regression model since we can see that the trend doesn't seem to follow a linear model. 

Estimating the number of deaths and ICU patients for the US:

![image](https://github.com/user-attachments/assets/a3479552-b47e-4b2e-bceb-2605fbc9f410)

![image](https://github.com/user-attachments/assets/7f3172d0-141c-4b09-beed-5df15f0af117)

From this graph, we can see that before the vaccine came out, there were many people placed into icu treatments. As the vaccines began to roll out, the number of icu patients significantly decreased with a slight peak. This may be due to new variants such as the delta variant, along with less strict lockdown regulations causing the vaccine to lose its effectiveness as well as causing more elderly and sick people to require icu treatment.

![image](https://github.com/user-attachments/assets/f99dd602-4cdd-44f4-b265-084270023c01)

This model doesn't make sense since we see the forecasted number of deaths lower than the last recorded amount of deaths. This model can be fine-tuned by using a different regression model. We can also add a constraint such that the first value should coincide with the last recorded amount of deaths.

![image](https://github.com/user-attachments/assets/005d74fb-8133-458f-8ea2-cf873fd95480)

We can see that the forecasted amount of icu patients is higher. This model can be fine-tuned by using a different regression model since we can see that the trend doesn't seem to follow a linear model.

**Part 5 - Deriving Insights about the effects of vaccination and discussion**

I compared Canada with its neighboring country, the United States. 

In terms of which country has the most effective vaccination program, it seems that currently, Canada is doing better in terms of vaccination rate with 79.21% of the population being vaccinated compared to the US vaccination rate of 68.5%. 

**Canada**:
I used the ARIMA model to forecast the vaccination rate after 50 days which will be 80.79% (a 1.58% increase). From our base case model, we can predict that 30,770,417 people will be vaccinated in Canada by the end of the next 50 days (from the last day of the dataset). This means 618,211 more people will be vaccinated.

**US**:
I used the ARIMA model to forecast the vaccination rate after 50 days which will be 71.57% (a 3.07% increase, almost double that as of Canada's).
From our base case model, we can predict that 240,826,270 people will be vaccinated in the US by the end of the next 50 days (from the last day of the dataset). This means 10,527,526 more people will be vaccinated.

There may be many factors or features affecting the vaccination rate in a country. This simple time series analysis by ARIMA forecasting equation is only considering a linear relationship between the days and vaccination rate in which the predictors comprised of lags of the dependent feature (here vaccination rate) and lags of the forecasting errors.

**Amount of Deaths vs Amount of People Vaccinated**: 
Plotting the amount of deaths due to COVID-19 vs. the amount of people vaccinated, I can observe that the Canadian trend is very similar to the US trend. We can see that before the vaccine came out, there were many deaths related to COVID-19. This is because it took a while for the vaccine rollout to occur. As vaccines progressively came out, the rate of deaths decreased, we can see the rate of deaths decreasing (flattening the curve). Lastly, the slope of deaths is exponentially increasing as of now, this may be due to many factors, such that new variants are out along with less strict lockdown regulations causing the vaccine to lose its effectiveness.

Canada has 29,533 deaths with a population of 38066161

The US has 770,691 deaths with a population of 332915074

US/ Canada Population = 8.74

US / Canada Deaths = 26.09

From this, we can observe that the US only has a population size of 8.74x greater than Canada yet 26.09x more deaths due to covid. This may be due to a lack of healthcare in the US. This may also be because it has a lower vaccination rate than in Canada and a cultural difference between freedom and social distancing and lockdown protocols. 

**Number of icu patients vs Amount of People Vaccinated**: 
Plotting the amount of icu patients vs the number of people vaccinated, I can observe that the Canadian trend is somewhat similar to the US trend.
From this graph, we can see that before the vaccine came out, there were many people placed into icu treatments. As the vaccines began to roll out, the number of icu patients significantly decreased with a slight peak. This may be due to new variants such as the delta variant, along with less strict lockdown regulations causing the vaccine to lose its effectiveness as well as causing more elderly and sick people to require icu treatment.

Canada has 468 icu patients with a population of 38066161

US has 11,315 icu patients with a population of 332915074

US/ Canada Population = 8.74

US / Canada icu patients = 24.17

From this, we can observe that the US only has a population size of 8.74x greater than Canada yet 24.17x icu_patients. It seems that getting covid in the US is much more serious and life-threatening compared to Canada. This may also be because it has a lower vaccination rate than in Canada and a cultural difference between freedom and social distancing and lockdown protocols. 

Canada has a much more effective vaccination program than the US based on the number of deaths and icu patients, as Canada has much fewer deaths and icu patients relative to the population size. 








