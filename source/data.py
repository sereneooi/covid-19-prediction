import pandas as pd

'''
path = os.getcwd() # get current path
vaccine_path = f'{path}\\citf-public\\vaccination\\vax_state.csv'
cases_path = f'{path}\\covid19-public\\epidemic\\cases_state.csv'

# get all the file in this path
cases_files = os.listdir(f'{cases_path}\\')

# get only the CSV files (.csv)
cases_files_csv = [f for f in cases_files if '.csv' in f] 
'''

# get the data from github and store into pandas dataframes
vaccine_data = pd.read_csv("https://raw.githubusercontent.com/CITF-Malaysia/citf-public/main/vaccination/vax_state.csv",
usecols = ['date', 'state', 'daily', 'cumul', 'cumul_full', 'pfizer1', 'sinovac1', 'astra1', 'sinopharm1', 'cansino'])
cases_data = pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_state.csv",
usecols = [0, 1, 2, 4, 5, 7, 8])
deaths_data = pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/deaths_state.csv", 
usecols = [0, 1, 2, 6, 7])
tests_data = pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/tests_state.csv")
linelist_deaths = pd.read_csv("https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/linelist/linelist_deaths.csv",
usecols = ['state', 'brand1', 'brand2'])


# filter only for Penang data
vaccine_data = vaccine_data[vaccine_data['state'] == "Pulau Pinang"]
cases_data = cases_data[cases_data['state'] == "Pulau Pinang"]
deaths_data = deaths_data[deaths_data['state'] == "Pulau Pinang"]
tests_data = tests_data[tests_data['state'] == "Pulau Pinang"]
linelist_deaths = linelist_deaths[linelist_deaths['state'] == "Pulau Pinang"]

# convert datetime
vaccine_data['date'] = pd.to_datetime(vaccine_data['date'], format = "%Y-%m-%d")
cases_data['date'] = pd.to_datetime(cases_data['date'], format = "%Y-%m-%d")
deaths_data['date'] = pd.to_datetime(deaths_data['date'], format = "%Y-%m-%d")
tests_data['date'] = pd.to_datetime(tests_data['date'], format = "%Y-%m-%d")

# remove the state column 
del vaccine_data["state"]
del cases_data["state"]
del deaths_data["state"]
del tests_data["state"]
del linelist_deaths["state"]

# merge all the data into single file, should be donw in the data.py
# join the data based on the datetime
# set cases_new to target variable, then the rest is independent variables
cases_data = pd.merge(cases_data, tests_data, on = 'date', how = 'outer')
cases_data = pd.merge(cases_data, vaccine_data, on = 'date', how = 'outer')
cases_data = pd.merge(cases_data, deaths_data, on = 'date', how = 'outer')

# store into a new csv file
#vaccine_data.to_csv("Vaccination.csv", index = False)
cases_data.to_csv("Cases.csv", index = False)
#deaths_data.to_csv("Deaths.csv", index = False)
#tests_data.to_csv("Tests.csv", index = False)
linelist_deaths.to_csv("linelist_deaths.csv", index = False)

'''
# prediction for future days
def predict(num_prediction, model):
    prediction_list = df.iloc[-num_prediction-1:, :].copy()
    
    for i in range(num_prediction):
        x = df.iloc[-win_length-1:, :]
        y = df.iloc[-win_length-1:, 0]
        num_rows, num_cols = x.shape
        x = TimeseriesGenerator(x, y, length = win_length, sampling_rate = 1, batch_size = batch_size)
        out = model.predict(x)
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[-num_prediction-1:]
        
    return prediction_list
    
def predict_dates(num_prediction):
    last_date = df.index.values[-1]
    prediction_dates = pd.date_range(last_date, periods = num_prediction + 1).tolist()

    return prediction_dates

num_prediction = 14
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)

print(len(forecast))

fct = pd.DataFrame(forecast_dates, columns = ['dates'])
fct['pred'] = forecast.tolist()
fct.index = pd.to_datetime(fct['dates'])
del fct["dates"] 

# plot predicted cases for future days
plt.title("Predicted New Daily Cases")
#df['cases_new'].plot(label = 'Actual New Daily Cases') 
fct['pred'].plot(label = 'Predicted New Daily Cases') 
plt.ylabel("Cases") # label y-axis
plt.legend()
plt.show()
'''