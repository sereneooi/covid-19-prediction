import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.dates as mdates


'''
# get the population of Penang
population = pd.read_csv("https://raw.githubusercontent.com/CITF-Malaysia/citf-public/main/static/population.csv", usecols = ['state', 'pop'])
population = int(population[population['state'] == "Pulau Pinang"]['pop'])
'''
# read data from csv files
data = pd.read_csv("Cases.csv").iloc[41:, :] # get the data after 2020-03-06
#data.set_index('date') # set date as index
data.index = pd.to_datetime(data['date'])
del data["date"] # delete the column 'date' since the date is set to index

# data cleansing
# all value cannot be negative
data.fillna(value = 0, inplace = True) # fill all NaN with 0

# convert variables from float to integer
data['cases_pvax'] = data['cases_pvax'].astype(np.int64)
data['cases_fvax'] = data['cases_fvax'].astype(np.int64)
data['rtk-ag'] = data['rtk-ag'].astype(np.int64)
data['pcr'] = data['pcr'].astype(np.int64)
data['daily'] = data['daily'].astype(np.int64)
data['cumul_full'] = data['cumul_full'].astype(np.int64)
data['cumul'] = data['cumul'].astype(np.int64)
data['deaths_new'] = data['deaths_new'].astype(np.int64)
data['deaths_pvax'] = data['deaths_pvax'].astype(np.int64)
data['deaths_fvax'] = data['deaths_fvax'].astype(np.int64)
data['pfizer1'] = data['pfizer1'].astype(np.int64)
data['sinovac1'] = data['sinovac1'].astype(np.int64)
data['astra1'] = data['astra1'].astype(np.int64)
data['sinopharm1'] = data['sinopharm1'].astype(np.int64)
data['cansino'] = data['cansino'].astype(np.int64)

# 1. features selection
# chi sqaure test
# Heap Map

'''
def checkOutlier():

# check if any negative values in the data
if (data.iloc[: ,1:].values < 0).any():
    checkOutlier()
'''

def showTrends(title, x_variable, y_variable):
    plt.title(title)
    plt.plot(x_variable, y_variable)
    X = plt.gca().xaxis # decalared X variable to x-axis
    X.set_major_locator(mdates.MonthLocator()) # set every month
    X.set_major_formatter(mdates.DateFormatter('%b %Y')) # Specify the format of date to be displayed
    plt.xticks(rotation = 45) # rotate the x-axis label so that can be seen clearly
    plt.ylabel("Cases") # label y-axis
    plt.show()

def calculateVaccineTypeForDeath():
    deaths = pd.read_csv("linelist_deaths.csv")
    deaths.fillna(value = 'NaN', inplace = True)
    p = 0
    s = 0
    az = 0
    c = 0
    for i in range(len(deaths)):
        if deaths['brand1'].iloc[i].lower() == 'sinovac':
            s += 1
        elif deaths['brand1'].iloc[i].lower() == 'pfizer':
            p += 1
        elif deaths['brand1'].iloc[i].lower() == 'astrazeneca':
            az += 1
        elif deaths['brand1'].iloc[i].lower() == 'cansino':
            c += 1

    return p, s, az, c

def calculateVaxDeathRate(vaxType, total_vaccinated): # calculate the number of deaths rate (deaths number) for each type of vaccine
    return vaxType / total_vaccinated * 100

def calculateVaccineType(vaxType): 
    vaxType.fillna(value = 0, inplace = True)
    total = 0
    
    for i in range(len(vaxType)):
        total += vaxType.iloc[i]

    return total

# used to display in the line graph
smooth_data = data.rolling(7).mean().round(5)

'''
# show all the trends
# 2021-02-24 is the date of first day of vaccination
showTrends("Daily Vaccination Trend in Penang", smooth_data.index[(data.index >= '2021-02-24')], smooth_data['daily'].loc[(data.index >= '2021-02-24')])
showTrends("Daily New Cases in Penang", smooth_data.index, smooth_data['cases_new'])
showTrends("Daily Active Cases in Penang", smooth_data.index, smooth_data['cases_active'])
# first death case happened on 2020-03-22
showTrends("Daily New Deaths in Penang", smooth_data.index[(data.index >= '2020-03-22')], smooth_data['deaths_new'].loc[(data.index >= '2020-03-22')])

# plot bars in stack manner
plt.title("Daily Test Cases in Penang")
plt.bar(smooth_data.index[(data['rtk-ag'] > 0)], smooth_data['rtk-ag'].loc[(data['rtk-ag'] > 0)], label = 'RTK-AG', color = (0.2, 0.4, 0.65, 0.8))
plt.bar(smooth_data.index[(data['pcr'] > 0)], smooth_data['pcr'].loc[(data['pcr'] > 0)], bottom = smooth_data['rtk-ag'].loc[(data['rtk-ag'] > 0)], label = 'PCR', 
color = (0.2, 0.4, 0.6, 0.6))
plt.legend()
X = plt.gca().xaxis # decalared X variable to x-axis
X.set_major_locator(mdates.MonthLocator()) # set every month
X.set_major_formatter(mdates.DateFormatter('%b %Y')) # Specify the format of date to be displayed
plt.xticks(rotation = 45) # rotate the x-axis label so that can be seen clearly
plt.ylabel("Cases") # label y-axis
plt.show()
'''

# calculate new cases rate




# calculate culmulative deaths amount
cumul_deaths = 0
for i in range(len(data)):
    cumul_deaths += data['deaths_new'].iloc[i]
print(f'Cumulative Deaths: {cumul_deaths}')

#print('Deaths Rate: {:.2f}'.format(data['deaths_rate']))

## calculate effectiveness of each vaccine
p, s, az, c = calculateVaccineTypeForDeath()

total_vaccinated_deaths = p + s + az + c
print(f'Total Vaccinated Deaths: {total_vaccinated_deaths}')


# calculate total number vaccinated of each vaccine type
total_p = calculateVaccineType(data['pfizer1'])
total_s = calculateVaccineType(data['sinovac1'])
total_az = calculateVaccineType(data['astra1'])
total_si = calculateVaccineType(data['sinopharm1'])
total_c = calculateVaccineType(data['cansino'])

print(total_p)
print(total_s)
print(total_az)
print(total_c)
print(total_si)

# calculate the deaths rate (possibility of death) of each vaccine type
p_death_rate = calculateVaxDeathRate(p, total_p)
s_death_rate = calculateVaxDeathRate(s, total_s)
az_death_rate = calculateVaxDeathRate(az, total_az)
c_death_rate = calculateVaxDeathRate(c, total_c)

print("Pfizer       : {:.2f} %".format(p_death_rate))
print("Sinovax      : {:.2f} %".format(s_death_rate))
print("Astra Zeneca : {:.2f} %".format(az_death_rate))
print("Casino       : {:.2f} %".format(c_death_rate))

# Test prediction result by using confusion matrix

# create new dataframe only store for the selected columns
#ds = smooth_data[['cases_new', 'cases_recovered', 'cases_pvax', 'cases_fvax', 'pcr', 'daily', 'deaths_new']].copy()
ds = smooth_data[['cases_new', 'cases_recovered', 'cases_pvax', 'cases_fvax', 'daily', 'deaths_new']].copy()
ds.fillna(value = 0, inplace = True)

# Visualize the trends in data together
sns.set_style('darkgrid')
ds.plot(kind = 'line', legend = 'reverse', title = 'Time-Series')
plt.legend(loc = 'upper right', shadow = True, bbox_to_anchor = (1.35, 0.8))
plt.legend()
plt.show()

# Splitting the dataset into train & test subsets
n_obs = 30
ds_train, ds_test = ds[:-n_obs], ds[-n_obs:]

# Augmented Dickey-Fuller Test (ADF Test) to check for stationarity
from statsmodels.tsa.stattools import adfuller

def adf_test(ds):
    dftest = adfuller(ds, autolag='AIC')
    adf = pd.Series(dftest[0:4], index = ['Test Statistic','p-value','# Lags','# Observations'])

    for key, value in dftest[4].items():
       adf['Critical Value (%s)'%key] = value
    print (adf)

    p = adf['p-value']
    if p <= 0.05:
        print("\nSeries is Stationary")
    else:
        print("\nSeries is Non-Stationary")


for i in ds_train.columns:
    print("Column: ",i)
    print('--------------------------------------')
    adf_test(ds_train[i])
    print('\n')

# Differencing all variables to get rid of Stationarity
ds_differenced = ds_train.diff().dropna()

# Running the ADF test once again to test for Stationarity
for i in ds_differenced.columns:
    print("Column: ",i)
    print('--------------------------------------')
    adf_test(ds_differenced[i])
    print('\n')

# Now cols: 3, 5, 6, 8 are non-stationary
ds_differenced = ds_differenced.diff().dropna()

# Running the ADF test for the 3rd time to test for Stationarity
for i in ds_differenced.columns:
    print("Column: ",i)
    print('--------------------------------------')
    adf_test(ds_differenced[i])
    print('\n')

# Fitting the VAR model to the 2nd Differenced Data
from statsmodels.tsa.api import VAR

model = VAR(ds_differenced)
results = model.fit(maxlags = 15, ic = 'aic')
results.summary()

# Forecasting for 100 steps ahead
lag_order = results.k_ar
predicted = results.forecast(ds_differenced.values[-lag_order:], n_obs)
forecast = pd.DataFrame(predicted, index = ds.index[-n_obs:], columns = ds.columns)

# Plotting the Forecasted values
p1 = results.plot_forecast(1)
p1.tight_layout()

# Inverting the Differencing Transformation
def invert_transformation(ds, df_forecast, second_diff=False):
    for col in ds.columns:
        # Undo the 2nd Differencing
        if second_diff:
            df_forecast[str(col)] = (ds[col].iloc[-1] - ds[col].iloc[-2]) + df_forecast[str(col)].cumsum()

        # Undo the 1st Differencing
        df_forecast[str(col)] = ds[col].iloc[-1] + df_forecast[str(col)].cumsum()

    return df_forecast

forecast_values = invert_transformation(ds_train, forecast, second_diff=True)

# Actual vs Forecasted Plots
fig, axes = plt.subplots(nrows = int(len(ds.columns)/2), ncols = 2, dpi = 100, figsize = (10,10))

for i, (col,ax) in enumerate(zip(ds.columns, axes.flatten())):
    forecast_values[col].plot(color = '#F4511E', label = f"Forecast {col}" , ax = ax).autoscale(axis ='x',tight = True)
    ds_test[col].plot(color = '#3949AB', ax = ax)

    ax.set_title(col + ' - Actual vs Forecast')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.legend()
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize = 6)

plt.tight_layout()
plt.savefig('actual_forecast.png')
plt.show()

# MSE
from sklearn.metrics import mean_squared_error
from numpy import asarray as arr
mse = mean_squared_error(ds_test, forecast_values)