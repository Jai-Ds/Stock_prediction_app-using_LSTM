import os,stat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pytz import timezone
from urllib import request
from datetime import timedelta


print("Hi started and waiting")
#time.sleep(300)

# %%
os.chdir("..")
dir=os.getcwd()
print(dir)



os.chmod("Nifty_100.csv", stat.S_IRWXU)
os.chmod("Nifty_50.csv", stat.S_IRWXU)
os.chmod("forecast_nifty50.csv",stat.S_IRWXU)
os.chmod("forecast_nifty100.csv", stat.S_IRWXU)


# %%
os.remove("Nifty_100.csv")
os.remove("forecast_nifty100.csv")

# %%
today = datetime.today()
now=today.strftime('%Y%m%d')

tomo=datetime.today() + timedelta(1)
nextday=tomo.strftime('%Y%m%d')

# %%
url='https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=CTSH&interval=60min&apikey=XYMY4ESJJ610R4KS&datatype=csv'

# %%
def download_stock_data(csv_url):
    response=request.urlopen(csv_url,)
    csv=response.read()
    csv_str=str(csv)
    lines=csv_str.split("\\n")
    dest_url=r'Nifty_100.csv'
    fx=open(dest_url,'w')
    for line in lines:
        fx.write(line + '\n')
    fx.close()

# %%
download_stock_data(url)

# %%
df=pd.read_csv('Nifty_100.csv')

# %%
df=df.rename(columns={'b\'timestamp': 'timestamp'})

# %%
"""
# Data Preprocessing
"""

# %%
def data_preprocessing(df):
    df=df.dropna()
    df=df.drop(labels=['open','high','low','volume\\r'],axis=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'],  errors='coerce')
    df['timestamp']=df['timestamp'].apply(lambda x:x.tz_localize('EST'))
    df['timestamp']=df['timestamp'].apply(lambda x:x.astimezone(timezone('Asia/Kolkata')))
    df['timestamp']=df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['timestamp'] = pd.to_datetime(df['timestamp'],  errors='coerce')
    df.drop(index=[99,98],axis=1,inplace=True)
    df.set_index(keys='timestamp',inplace=True)
    df.columns=['Price']
    df=df.sort_index(ascending=True)
    return df

# %%
df=data_preprocessing(df)

# %%
from sklearn.preprocessing import MinMaxScaler

# %%
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

# %%
n_features=1

# %%
"""
# Forecasting
"""

# %%
full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)

# %%
length = 7 # Length of the output sequences (in number of timesteps)
generator = TimeseriesGenerator(scaled_full_data, scaled_full_data, length=length, batch_size=1)

# %%
# define model
model = Sequential()

# Simple RNN layer
model.add(LSTM(7,input_shape=(length, n_features),return_sequences=True))
model.add(LSTM(7))


# Final Prediction
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# %%
model.fit_generator(generator,epochs=8)

# %%
forecast = []
# Replace periods with whatever forecast length you want
periods = 7

first_eval_batch = scaled_full_data[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(periods):

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]

    # store prediction
    forecast.append(current_pred)

    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

# %%
forecast = full_scaler.inverse_transform(forecast)

# %%
labels=["10:15","11:15","12:15","1:15","2:15","3:15","4:15"]

# %%
forecast_df = pd.DataFrame(data=forecast,index=labels,
                           columns=['Forecast'])

# %%
forecast_df

# %%
forecast_df.to_csv('forecast_nifty100.csv', index=True)

# %%
forecast_df.plot(figsize=(12,8))
plt.savefig(dir+'/static/images/Nifty100_hourly_forecast_'+now+'.png')

# %%
"""

# --------------------------------------------------------------------------------------------------------------
"""

# %%
print("NSE Done")

# %%
"""

# --------------------------------------------------------------------------------------------------------------
"""

# %%
os.remove("Nifty_50.csv")
os.remove("forecast_nifty50.csv")

# %%
url='https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=ibm&interval=60min&apikey=XYMY4ESJJ610R4KS&datatype=csv'

# %%
def download_stock_data1(csv_url):
    response=request.urlopen(csv_url,)
    csv=response.read()
    csv_str=str(csv)
    lines=csv_str.split("\\n")
    dest_url=r'Nifty_50.csv'
    fx=open(dest_url,'w')
    for line in lines:
        fx.write(line + '\n')
    fx.close()

# %%
download_stock_data1(url)


# %%
df=pd.read_csv('Nifty_50.csv')

# %%
df=df.rename(columns={'b\'timestamp': 'timestamp'})

# %%
df=data_preprocessing(df)

# %%
"""
# Scaling
"""

# %%
full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)

# %%
length = 7 # Length of the output sequences (in number of timesteps)
generator = TimeseriesGenerator(scaled_full_data, scaled_full_data, length=length, batch_size=1)

# %%
n_features = 1

# %%
"""
# Model
"""

# %%
model = Sequential()

# Simple  layer
model.add(LSTM(7,input_shape=(length, n_features),return_sequences=True))
model.add(LSTM(7))

# Final Prediction
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# %%
model.fit_generator(generator,epochs=8)

# %%
"""
# Forecast
"""

# %%
forecast = []
# Replace periods with whatever forecast length you want
periods = 7

first_eval_batch = scaled_full_data[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(periods):

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]

    # store prediction
    forecast.append(current_pred)

    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

# %%
forecast = full_scaler.inverse_transform(forecast)

# %%
forecast_df = pd.DataFrame(data=forecast,index=labels,
                           columns=['Forecast'],)

# %%
forecast_df

# %%
forecast_df.to_csv('forecast_nifty50.csv', index=True)

# %%
forecast_df.plot(figsize=(12,8))
plt.savefig(dir+'/static/images/Hourly_Forcast_Nifty50_'+now+'.png')

# %%
print("All Done")