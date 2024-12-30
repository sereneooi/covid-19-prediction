import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.dates as mdates
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler


'''
# get the population of Penang
population = pd.read_csv("https://raw.githubusercontent.com/CITF-Malaysia/citf-public/main/static/population.csv", usecols = ['state', 'pop'])
population = int(population[population['state'] == "Pulau Pinang"]['pop'])
'''
# read data from csv files
data = pd.read_csv("Cases.csv").iloc[41:, :] # get the data after 2020-03-06

# convert all columns to float64 except for date column
cols = data.columns.drop('date')
data[cols] = data[cols].apply(np.float64)
data.index = pd.to_datetime(data['date']) # set date column to index
del data["date"] # delete the column 'date' since the date is set to index

# data cleansing
# all value cannot be negative
data.fillna(value = 0, inplace = True) # fill all NaN with 0

# 1. features selection
# chi sqaure test
# Heap Map

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

print("Pfizer       : {:.3f} %".format(p_death_rate))
print("Sinovax      : {:.3f} %".format(s_death_rate))
print("Astra Zeneca : {:.3f} %".format(az_death_rate))
print("Casino       : {:.3f} %".format(c_death_rate))

# Test prediction result by using confusion matrix

def predict(p, s, az):
    new_data = data[['cases_new', 'deaths_new', 'pfizer1', 'sinovac1', 'astra1']].copy()
    predicted_pfizer = []
    predicted_sinovax = []
    predicted_az = []
    total_predicted = []
    for i in range(len(new_data)):
        predicted_pfizer.append(float(new_data['pfizer1'].iloc[i] * p))
        predicted_sinovax.append(float(new_data['sinovac1'].iloc[i] * s))
        predicted_az.append(float(new_data['astra1'].iloc[i] * az))
        total_predicted.append(predicted_pfizer[i]+predicted_sinovax[i]+predicted_az[i])
    
    new_data['total_predicted'] = total_predicted

    print(new_data['total_predicted'])

    plt.title("Predicted")
    new_data['total_predicted'].plot(label = 'Predicted')
    new_data['cases_new'].plot(label = 'Actual Cases')
    X = plt.gca().xaxis # decalared X variable to x-axis
    X.set_major_locator(mdates.MonthLocator()) # set every month
    X.set_major_formatter(mdates.DateFormatter('%b %Y')) # Specify the format of date to be displayed
    plt.xticks(rotation = 45) # rotate the x-axis label so that can be seen clearly
    plt.ylabel("Cases") # label y-axis
    plt.legend()
    plt.show()

# create new dataframe only store for the selected columns
#ds = smooth_data[['cases_new', 'cases_recovered', 'cases_pvax', 'cases_fvax', 'pcr', 'daily', 'deaths_new']].copy()
ds = smooth_data[['cases_new', 'cases_recovered', 'cases_pvax', 'cases_fvax', 'daily', 'deaths_new']].copy()
ds.fillna(value = 0, inplace = True)

predict(p_death_rate, s_death_rate, az_death_rate)
