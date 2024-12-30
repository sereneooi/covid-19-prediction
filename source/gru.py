import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import GRU
import tensorflow as tf
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# Build Model
def GRU_model(X_Train, y_Train, X_Test):
    early_stopping = tf.keras.callbacks.EarlyStopping(min_delta = 0.001, patience = 10, restore_best_weights = True) 
    
    # GRU model 
    model = Sequential()
    model.add(GRU(units=150, return_sequences=True, input_shape=(X_Train.shape[1],1), activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(GRU(units=150, return_sequences=True, input_shape=(X_Train.shape[1],1), activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(GRU(units=50, return_sequences=True, input_shape=(X_Train.shape[1],1), activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(GRU(units=50, return_sequences=True, input_shape=(X_Train.shape[1],1), activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(GRU(units=50, input_shape=(X_Train.shape[1], 1), activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=1))

    # Compiling the model
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = [tf.metrics.MeanAbsoluteError()])
    model.fit(X_Train, y_Train, epochs = 100, batch_size = 200, callbacks = [early_stopping])
    pred_GRU = model.predict(X_Test)

    # save model
    model.save('gru.h5')

    return pred_GRU

# To calculate the root mean squred error in predictions
def RMSE_Value(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))
    
    return rmse

# To plot the comparitive plot of targets and predictions
def PredictionsPlot(test, predicted, m):
    plt.figure(figsize = (12,5), facecolor = "#627D78")
    plt.plot(test, color=colours[m], label="Actual Value", alpha=0.5 )
    plt.plot(predicted, color="#627D78", label="Predicted Value")
    plt.title("GRU Prediction Vs Actual New Cases")
    plt.xlabel("Date")
    plt.ylabel("New Cases")
    plt.legend()
    plt.show()

# read data from csv files
data = pd.read_csv("Cases.csv", parse_dates = True).iloc[41:, :] # get the data after 2020-03-06
data.set_index('date')

# convert all columns to float64 except for date column
cols = data.columns.drop('date')
data[cols] = data[cols].apply(np.float64)
data.index = pd.to_datetime(data['date'])   # set date column to index
del data["date"]                            # delete the column 'date' since the date is set to index
data.fillna(value = 0, inplace = True)      # fill all NaN with 0

# df to be used for EDA
df = data.copy()

# Let's plot the Timeseries
colors = ['Accent', 'Accent_r', 'Blues', 'Blues_r']
plt.figure(figsize = (20,4), facecolor = "#627D78")
Time_series = sns.lineplot(x = df.index, y = df['cases_new'], data = df, palette = colors[2])
Time_series.set_title("Covid-19 in Penang")
Time_series.set_ylabel("Number of Daily New Cases")
Time_series.set_xlabel("Date")

# subplot
x = df.index
y = df.columns

colours = ['teal', 'darkturquoise', 'cadetblue', 'powderblue', 'lightblue',
           'deepskyblue', 'steelblue', 'lightskyblue', 'dodgerblue', 'darkslategrey', 
           'royalblue', 'paleturquoise', 'mediumturquoise', 'forestgreen', 'darkseagreen', 
           'green', 'seagreen', 'limegreen']

plt.figure(figsize = (25, 45), facecolor = "#627D78")

j = 0

for i in range(len(y)):
    plt.subplot(6, 3, i+1)
    plt.plot(x, df[y[i]], color = colours[j])
    plt.title('Daily ' + y[i])
    plt.xlabel('date')
    plt.ylabel(y[i])
    X = plt.gca().xaxis                         # decalared X variable to x-axis
    X.set_major_locator(mdates.YearLocator())   # set every year
    X.set_major_formatter(mdates.DateFormatter('%b %Y')) # Specify the format of date to be displayed
    j += 1

plt.show()

# check heatmap
corr = df.corr()
plt.subplots(figsize = (20, 20), facecolor = "#627D78")
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, cmap = "Blues", annot = True, linewidths = .5)
plt.show()

c = data.corr().abs()
s = c.unstack()
sort = s.sort_values(kind = "quicksort")
print(sort)

target = df['cases_new']
features = df.drop('cases_new', 1)  # 1 = cols, 0 = rows

# shift the value to the previous 14 days
features = features.shift(periods = -14, fill_value = 0)

df = pd.concat([target, features], axis = 1)

# heatmap
corr = df.corr()
plt.subplots(figsize = (20, 20), facecolor = "#627D78")
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, cmap = "Blues", annot = True, linewidths = .5)
plt.show()

# features selection
# cases_recovered, cases_active, cases_pvax, deaths_new
df = df[['cases_new', 'cases_recovered', 'cases_active', 'cases_pvax', 'deaths_new']]

# see the relationship
sns.pairplot(data = df)

# normalized the data 
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

features = data_scaled
target = data_scaled[:, 0]
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 123, shuffle = False)

# Predictions
pred = GRU_model(x_train, y_train, x_test)

RMSE = RMSE_Value(y_test, pred)
PredictionsPlot(y_test, pred, 0)

# Inverse the value back to normal
df_pred = pd.concat([pd.DataFrame(pred), pd.DataFrame(x_test[:, 1:])], axis = 1)
rev_trans = scaler.inverse_transform(df_pred)
df_final = df[pred.shape[0]*-1:]

# store the predicted value into the new column
df_final['cases_new_pred'] = rev_trans[:, 0] # 0 means the index, which is cases_new

# plot 
plt.figure(figsize = (25, 15), facecolor = "#627D78")
plt.title("Predicted New Daily Cases")
df_final['cases_new_pred'].plot(label = 'Predicted New Daily Cases', color = colours[7])
# df_final will only display the actual cases that is being used by the prediction, if want show all then change to df instead of df_final
df['cases_new'].plot(label = 'Actual New Daily Cases', color = colours[2]) 
plt.ylabel("Cases") # label y-axis
plt.legend()
plt.show()