import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import datetime
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


def DataPreprcess(fx_pair, interval):

    # set the data range for getting data
    end_date = datetime.datetime.now()
    if (interval == 'd'):
        x = datetime.timedelta(days=80)
        end_date = end_date - datetime.timedelta(days=1)
    elif (interval == 'wk'):
        x = datetime.timedelta(weeks=80)
        end_date = end_date - datetime.timedelta(weeks=1)
    start_date = end_date - x

    # get data fom yahoo using data reader
    FX = yf.download(fx_pair+'=x',
                     start=start_date,
                     end=end_date,
                     progress=False,
                     interval="1"+interval)  # 1wk 1d
    currency_data = FX.filter(['Adj Close'])
    currency_data = currency_data[:len(currency_data)]

    # get the last 30 day price values and convert the dataframe to an array
    last_30 = currency_data[-30:].values.tolist()

    # scale the data to be values between 0 and 1
    sc = MinMaxScaler(feature_range=(0, 1))
    last_30_scaled = sc.fit_transform(last_30)

    # create an empty list
    X_test = []

    # append the past 30 days
    X_test.append(last_30_scaled)

    # convert the X_test data set to a numpy array
    X_test = np.array(X_test)

    # reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_test, sc, start_date, end_date


def DefineInterval(interval):
    if (interval == 'daily'):
        return 'd'
    elif (interval == 'weekly'):
        return 'wk'


app = FastAPI()

# cors
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# cors


@app.get('/')  # basic get view
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}


@app.get('/predict/{fx_pair}/{interval}')
async def read_user_item(fx_pair: str, interval: str):
    timeframe = DefineInterval(interval)

    if (timeframe == 'd'):
        if (fx_pair == 'eurusd'):
            loaded_model = tf.keras.models.load_model(
                'EURUSD_Daily.h5')
        elif (fx_pair == 'usdjpy'):
            loaded_model = tf.keras.models.load_model(
                'USDJPY_Daily.h5')
    elif (timeframe == 'wk'):
        if (fx_pair == 'eurusd'):
            loaded_model = tf.keras.models.load_model(
                'EURUSD_Weekly.h5')
        elif (fx_pair == 'usdjpy'):
            loaded_model = tf.keras.models.load_model(
                'USDJPY_Weekly.h5')

    X_test, sc, start_date, end_date = DataPreprcess(fx_pair, timeframe)
    predictions = loaded_model.predict(X_test)
    predictions = sc.inverse_transform(predictions)
    predictions = float("{:.5f}".format((predictions[0][0])))

    return {
        "pair": fx_pair,
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        "interval": interval,
        "prediction": predictions
    }
