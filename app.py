import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
start = '2010-01-01'
end = '2019-12-31'

st.title("Bit Coin Predictor")
user_input= st.text_input("Enter Stock Ticker",'AAPL')
df = yf.download(user_input, start=start, end=end)

st.subheader('Data from 2010 - 2019')
st.write(df.describe())

st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

#Splitting data into train and test
data_train=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test=pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_train_array=scaler.fit_transform(data_train)


# Load the model
model=load_model('LSTM_model.h5')

# Testing
past_100_days=data_train.tail(100)
final_df=past_100_days.append(data_test, ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test), np.array(y_test)
y_predict=model.predict(x_test)
scaler=scaler.scale_
scale_factor= 1/scaler[0]
y_predict= y_predict * scale_factor
y_test=y_test * scale_factor

# Predicted vs Original
st.subheader("Predicted vs Original")
fig_1=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predict, 'r', label= 'Predicted price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
st.pyplot(fig_1)
st.set_option('deprecation.showPyplotGlobalUse', False)