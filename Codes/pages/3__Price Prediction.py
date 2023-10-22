#streamlit run c:\Users\user\Desktop\Codes\Homepage.py

import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, Dropout 
from keras.models import Sequential
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from datetime import date, datetime
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, mean_squared_error, r2_score
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from tensorflow import keras
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os
from PIL import Image
import fundamentalanalysis as fa

st.set_option('deprecation.showPyplotGlobalUse', False)

###############################################################################################################

st.write(" ")
st.write(" ")
st.title("Stock Price Prediction - LSTM")


col1 = st.columns(1)

with col1[0].expander("Model Description", expanded=True):
    st.markdown("""
    Intrinsic Price Forecasting (IPF) is a predictive model that combines technical analysis techniques with the concept of intrinsic value to forecast future price movements of stocks. By leveraging historical price data, volatility measures, and directional indicators, the IPF model incorporates mean reversion principles while considering the inherent worth of the stock. It identifies patterns, trends, and potential deviations from the intrinsic value, allowing for predictions that reflect the expected direction and level of the stock's price based on its historical behavior. The IPF model provides insights into potential buying or selling opportunities by assessing the alignment of the stock's market price with its estimated intrinsic value. By integrating technical analysis and intrinsic value considerations, the IPF model aims to support informed investment decisions based on the projected future movement of stock prices.
    """)

    


# st.subheader("Technical Intrinsic Forecasting")
# subheader = ""
# st.markdown(f"<p style='font-family: sans-serif; font-size: 14px; font-style: italic;'>{subheader}</p>", unsafe_allow_html=True)
# subheader = 'Second chart refers to the weekly trend of the stock.'
# st.markdown(f"<p style='font-family: sans-serif; font-size: 14px; font-style: italic;'>{subheader}</p>", unsafe_allow_html=True)
# subheader = 'Third chart refers to the annual tenure of the stock.'
# st.markdown(f"<p style='font-family: sans-serif; font-size: 14px; font-style: italic;'>{subheader}</p>", unsafe_allow_html=True)

###############################################################################################################

def add_logo(logo_path, width, height):
    logo = Image.open("c:/Users/user/Desktop/Codes/Logo.png")
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path="c:/Users/user/Desktop/Codes/Logo.png", width=300, height=200)
st.sidebar.image(my_logo)

START = "2020-01-01"
END = "2023-02-01"
NEW = "2023-07-18"
FUTURE = "2024-02-01"

stocks = ("AAPL", "MSFT", "GME", "JNJ", "NVO")
if 'selected_stock' not in st.session_state:
    st.session_state['selected_stock'] = 'AAPL'

if 'n_years' not in st.session_state:
    st.session_state['n_years'] = 1

st.session_state['selected_stock'] = st.sidebar.selectbox("Select dataset for prediction", stocks, index=stocks.index(st.session_state['selected_stock']))

st.session_state['period'] = st.session_state['n_years'] * 365

selected_stock = st.session_state['selected_stock']
period = st.session_state['period']
n_years = st.session_state['n_years']

@ st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, END)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)


# Sidebar options

ticker = selected_stock
# Download stock data

df = yf.download(ticker, start=START, end=END).reset_index()

api_key = "ff8359f54188070791a9f0d88aaf1b41"

@st.cache_data
def profile(ticker, api_key):
        df_summary = fa.profile(ticker, api_key)
        return df_summary

df_summary = profile(selected_stock, api_key)

company_name = df_summary[0][8]

label1 = ""
value1 = company_name

# Add custom CSS styles
st.markdown("""
    <style>
        .fixed-panel {
            position: fixed;
            top: 0;
            width: 100%;
            background-color: #262730;
            text-color: #FFFFFF;
            padding: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            z-index: 999;
        }
        .fixed-panel h3 {
            color: white;
            padding-left: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Display the fixed panel with the title
st.markdown("""
    <div class="fixed-panel"><br></br>
        <h3><strong>{value1}</strong></h3>
    </div>
""".format(value1=value1), unsafe_allow_html=True)

#######################################################################################################

#Combine and average 4 componenets of stock for y-value
values = (df['High'] + df['Low'] + df['Open'] + df['Close'])/4
df = df.assign(Price=values)

df = df.drop('Open', axis=1)
df = df.drop('High', axis=1)
df = df.drop('Low', axis=1)
df = df.drop('Close', axis=1)
df = df.drop('Adj Close', axis=1)
df = df.drop('Volume', axis=1)

#Set index to Date
data = df.set_index('Date')

#Tie Date to Price
ts = data['Price']

########################################################################################################################################################

# Timeseries logged
ts_log = np.log(ts)

# Split the data into training and test sets
train_size = int(len(ts_log) * 0.8)
train_data = ts_log[:train_size]
test_data = ts_log[train_size:]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
test_scaled = scaler.transform(test_data.values.reshape(-1, 1))

########################################################################################################################################################

# Define the length of input sequences
sequence_length = 12

# Function to create input sequences
def create_sequences(ts, sequence_length):
    X = []
    y = []
    for i in range(len(ts) - sequence_length):
        X.append(ts[i:i+sequence_length])
        y.append(ts[i+sequence_length])
    return np.array(X), np.array(y)

# Create input sequences for training and testing
X_train, y_train = create_sequences(train_scaled, sequence_length)
X_test, y_test = create_sequences(test_scaled, sequence_length)

# Reshape the input data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

########################################################################################################################################################

# Define the LSTM model architecture 
model = Sequential()
model.add(LSTM(units=64, input_shape=(sequence_length, 1)))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=20, batch_size=50, validation_data=(X_test, y_test), callbacks=[early_stopping])

###############################################################################################################
###############################################################################################################
###############################################################################################################
### GRAPH 1

model = keras.models.load_model('Codes/AAPL.h5')

# Predict on the test data
y_pred_scaled = model.predict(X_test)

# Reverse the MinMax scaling for the predicted values
y_pred_inverse = scaler.inverse_transform(y_pred_scaled)
y_pred_inverse = np.exp(y_pred_inverse)

# Create an array of dates corresponding to the test data
dates_train = test_data.index
dates_test = train_data.index

# minus sequence length to date for prediction
y_pred_inverse_date = dates_train[:-sequence_length]

#Change date and prediction into same df

# Change date and prediction into the same DataFrame
y_pred_df = pd.DataFrame({'Date': y_pred_inverse_date, 'Price': [y_pred_inverse[i][0] for i in range(len(y_pred_inverse))]})

st.markdown("<hr>", unsafe_allow_html=True)

# Plot predictions vs actual values
st.subheader('Trained and Tested Model')
def plot_predictions_actual_values():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_test[:train_size], y=np.exp(train_data), name='Train',
                             line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=dates_train[:train_size], y=np.exp(test_data), name='Actual',
                             line=dict(color='firebrick')))
    fig.add_trace(go.Scatter(x=y_pred_inverse_date, y=y_pred_df['Price'], name='Predictions',
                             line=dict(color='orange')))
    fig.update_layout(xaxis_rangeslider_visible=True,
                      xaxis_title='Date',
                      yaxis_title='Price',
                      legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)

plot_predictions_actual_values()

# Reverse the MinMax scaling for the test data
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
y_test_inverse = np.exp(y_test_inverse)

# Calculate evaluation metrics
# mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
# mape = mean_absolute_percentage_error(y_test_inverse, y_pred_inverse)
# msle = mean_squared_log_error(y_test_inverse, y_pred_inverse)
rmse = mean_squared_error(y_test_inverse, y_pred_inverse, squared=False)
r2 = r2_score(y_test_inverse, y_pred_inverse)

rmse = round(rmse, 2)
r2 = r2*100
r2 = round(r2, 0)


# Display evaluation metrics
st.subheader('Evaluation Metrics')
st.write('Root Mean Squared Error (RMSE):', "$" + str(rmse))
st.write('R-squared:', str(r2) + "%")

st.markdown("<hr style='border: 0; height: 1px; background-color: #000; font-weight: strong;'>", unsafe_allow_html=True)

###############################################################################################################
###############################################################################################################
###############################################################################################################
### GRAPH 2

st.write("") 
st.write("")
st.write("") 

# Mapping of stocks to model file paths
model_files = {
    'AAPL': 'Codes/AAPL.h5',
    'GME': 'Codes/GME.h5',
    'JNJ': 'Codes/JNJ.h5',
    'NVO': 'Codes/NVO.h5',
    'MSFT': 'Codes/MSFT.h5'
    }

selected_stock = st.session_state['selected_stock']

# Check if the selected stock has a corresponding model file
if selected_stock in model_files:
    model_file = model_files[selected_stock]
    if os.path.exists(model_file):
        model = keras.models.load_model(model_file)
    else:
        st.sidebar.warning(f"Model file for {selected_stock} does not exist.")
else:
    st.sidebar.warning("No model file found for the selected stock.")


# Make future predictions from END to NEW
future_dates = pd.date_range(start=END, end=NEW).tolist()
future_data = test_scaled[-sequence_length:, :].reshape(1, sequence_length, 1)

predictions = []
for _ in range(len(future_dates)):
    prediction = model.predict(future_data)
    predictions.append(prediction[0][0])
    future_data = np.append(future_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

# Reverse the MinMax scaling for the predictions
predictions_scaled = np.array(predictions).reshape(-1, 1)
predictions_inverse = np.exp(scaler.inverse_transform(predictions_scaled))

# Create a DataFrame for future predictions
future_predictions = pd.DataFrame({'Date': future_dates, 'Price': predictions_inverse.flatten()})

###############################################################################################################

df = yf.download(ticker, start=START, end=NEW).reset_index()

# Combine and average 4 componenets of stock for y-value
values = (df['High'] + df['Low'] + df['Open'] + df['Close'])/4
df = df.assign(Price=values)

df = df.drop('Open', axis=1)
df = df.drop('High', axis=1)
df = df.drop('Low', axis=1)
df = df.drop('Close', axis=1)
df = df.drop('Adj Close', axis=1)
df = df.drop('Volume', axis=1)

#Set index to Date
data = df.set_index('Date')

#Tie Date to Price
ts = data['Price']

###############################################################################################################

# Plot predicted prices for future dates
st.subheader('Predicted Prices for Existing Dates')
def plot_future_predictions(): 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts, name='Actual',
                             line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=future_predictions['Date'], y=future_predictions['Price'], name='Predictions',
                             line=dict(color='orange')))
    fig.update_layout(xaxis_rangeslider_visible=True,
                      xaxis_title='Date',
                      yaxis_title='Price',
                      legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)

plot_future_predictions()


# Get the actual stock prices for the period from END to NEW
actual_prices = ts[END:NEW].values

# Get the predicted prices for the same period
predicted_prices = future_predictions['Price'].values[:len(actual_prices)]

def mean_error(actual_prices, predicted_prices):
    errors = [actual_prices[i] - predicted_prices[i] for i in range(len(actual_prices))]
    mean_error = sum(errors) / len(errors)
    return mean_error

me = mean_error(actual_prices, predicted_prices)
me = round(me, 2)

def mean_error_percentage(actual_prices, predicted_prices):
    errors = [(actual_prices[i] - predicted_prices[i]) / actual_prices[i] * 100 for i in range(len(actual_prices))]
    mean_error_percentage = sum(errors) / len(errors)
    return mean_error_percentage

me_percent = mean_error_percentage(actual_prices, predicted_prices)
me_percent = abs(me_percent)
me_percent = round(me_percent, 2)

# Display the evaluation metrics
st.subheader('Evaluation Metrics')
st.write('Mean Error (ME):', "$" + str(me))
st.write('Mean Error Percentage (ME%):', str(me_percent)  + "%")

st.markdown("<hr style='border: 0; height: 1px; background-color: #000; font-weight: strong;'>", unsafe_allow_html=True)

###############################################################################################################
###############################################################################################################
###############################################################################################################
### GRAPH 3

st.write("") 
st.write("")
st.write("") 

# Download stock data
df = yf.download(ticker, start=START, end=NEW).reset_index()

#Combine and average 4 componenets of stock for y-value
values = (df['High'] + df['Low'] + df['Open'] + df['Close'])/4
df = df.assign(Price=values)

df = df.drop('Open', axis=1)
df = df.drop('High', axis=1)
df = df.drop('Low', axis=1)
df = df.drop('Close', axis=1)
df = df.drop('Adj Close', axis=1)
df = df.drop('Volume', axis=1)

#Set index to Date
data = df.set_index('Date')

#Tie Date to Price
ts = data['Price']

########################################################################################################################################################

# Timeseries logged
ts_log = np.log(ts)

# Split the data into training and test sets
train_size = int(len(ts_log) * 0.8)
train_data = ts_log[:train_size]
test_data = ts_log[train_size:]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
test_scaled = scaler.transform(test_data.values.reshape(-1, 1))

########################################################################################################################################################

# Define the length of input sequences
sequence_length = 12

# Function to create input sequences
def create_sequences(ts, sequence_length):
    X = []
    y = []
    for i in range(len(ts) - sequence_length):
        X.append(ts[i:i+sequence_length])
        y.append(ts[i+sequence_length])
    return np.array(X), np.array(y)

# Create input sequences for training and testing
X_train, y_train = create_sequences(train_scaled, sequence_length)
X_test, y_test = create_sequences(test_scaled, sequence_length)

# Reshape the input data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

###############################################################################################################

# Mapping of stocks to model file paths
model_files = {
    'AAPL': 'Codes/AAPL_2.h5',
    'GME': 'Codes/GME_2.h5',
    'JNJ': 'Codes/JNJ_2.h5',
    'NVO': 'Codes/NVO_2.h5',
    'MSFT': 'Codes/MSFT_2.h5'
}

selected_stock = st.session_state['selected_stock']

# Check if the selected stock has a corresponding model file
if selected_stock in model_files:
    model_file = model_files[selected_stock]
    if os.path.exists(model_file):
        model = keras.models.load_model(model_file)
    else:
        st.sidebar.warning(f"Model file for {selected_stock} does not exist.")
else:
    st.sidebar.warning("No model file found for the selected stock.")


# Make future predictions from NEW to FUTURE
future_dates = pd.date_range(start=NEW, end=FUTURE).tolist()
future_data = test_scaled[-sequence_length:, :].reshape(1, sequence_length, 1)

predictions = []
for _ in range(len(future_dates)):
    prediction = model.predict(future_data)
    predictions.append(prediction[0][0])
    future_data = np.append(future_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

# Reverse the MinMax scaling for the predictions
predictions_scaled = np.array(predictions).reshape(-1, 1)
predictions_inverse = np.exp(scaler.inverse_transform(predictions_scaled))

# Create a DataFrame for future predictions
future_predictions = pd.DataFrame({'Date': future_dates, 'Price': predictions_inverse.flatten()})

###############################################################################################################

df = yf.download(ticker, start=START, end=NEW).reset_index()

#Combine and average 4 componenets of stock for y-value
values = (df['High'] + df['Low'] + df['Open'] + df['Close'])/4
df = df.assign(Price=values)

df = df.drop('Open', axis=1)
df = df.drop('High', axis=1)
df = df.drop('Low', axis=1)
df = df.drop('Close', axis=1)
df = df.drop('Adj Close', axis=1)
df = df.drop('Volume', axis=1)

#Set index to Date
data = df.set_index('Date')

#Tie Date to Price
ts = data['Price']

###############################################################################################################

# Plot predicted prices for future dates
st.subheader('Predicted Prices for Future Dates')
def plot_future_predictions():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts, name='Actual',
                             line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=future_predictions['Date'], y=future_predictions['Price'], name='Predictions',
                             line=dict(color='orange')))
    fig.update_layout(xaxis_rangeslider_visible=True,
                      xaxis_title='Date',
                      yaxis_title='Price',
                      legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)

plot_future_predictions()
