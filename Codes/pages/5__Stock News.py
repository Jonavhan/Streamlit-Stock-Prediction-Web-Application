# pip install streamlit requests

import streamlit as st
import requests
import streamlit as st
from datetime import date

st.write(" ")
st.write(" ")
st.title("Stock News Page")

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly as px

from PIL import Image

from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric

import pandas as pd
import fundamentalanalysis as fa


import pickle
from pathlib import Path 

import streamlit as st 
import streamlit_authenticator as stauth 

def add_logo(logo_path, width, height):
    logo = Image.open("c:/Users/user/Desktop/Codes/Logo.png")
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path="c:/Users/user/Desktop/Codes/Logo.png", width=300, height=200)
st.sidebar.image(my_logo)

# image = Image.open('c:/Users/user/Desktop/Codes/Image.png')
# st.image(image)

START = "2012-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

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


######################################################################################################

api_key = "ff8359f54188070791a9f0d88aaf1b41"

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)

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

######################################################################################################



def get_stock_news(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&language=en&apiKey=b22171f7278b49c5a2a25fdcae2358b5"
    response = requests.get(url)
    data = response.json()
    articles = data["articles"]
    return articles

selected_stock=st.session_state['selected_stock']

if selected_stock:
    articles = get_stock_news(selected_stock)
    if not articles:
        st.warning("No news found for the selected stock.")
    else:
        for article in articles:
            st.subheader(article["title"])
            st.write(f"Published at: {article['publishedAt']}")
            st.write(article["description"])
            st.markdown(f"Read more: [{article['source']['name']}]({article['url']})")
            st.markdown("---")




# import streamlit as st
# import requests

# def get_stock_news(symbol):
#     url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey=b22171f7278b49c5a2a25fdcae2358b5"
#     response = requests.get(url)
#     data = response.json()
#     articles = data["articles"]
#     return articles

# st.title("Stock News Page")
# symbol = st.text_input("Enter stock symbol (e.g., AAPL):")

# if symbol:
#     articles = get_stock_news(symbol)
#     if not articles:
#         st.warning("No news found for the given symbol.")
#     else:
#         for article in articles:
#             st.subheader(article["title"])
#             st.write(f"Published at: {article['publishedAt']}")
#             st.write(article["description"])
#             st.markdown(f"Read more: [{article['source']['name']}]({article['url']})")
#             st.markdown("---")
