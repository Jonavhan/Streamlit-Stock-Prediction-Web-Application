import streamlit as st

st.write(" ")
st.write(" ")
st.title("Trend Analysis - FB Prophet")


from datetime import date

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
import fundamentalanalysis as fa


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
# st.session_state['n_years'] = st.sidebar.slider("Years of prediction:", 1 , 4, value=st.session_state['n_years'])
st.session_state['period'] = st.session_state['n_years'] * 365

selected_stock = st.session_state['selected_stock']
period = st.session_state['period']
n_years = st.session_state['n_years']

#######################################################################################################

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

#######################################################################################################

#Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"})

m = Prophet(interval_width=0.95)
m.fit(df_train)
future = m.make_future_dataframe(periods = 365)
forecast = m.predict(future)

fig2 = m.plot_components(forecast)
axes = fig2.get_axes()

fig2.subplots_adjust(hspace=0.5)

axes[0].set_title('Trend Plot')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('')

axes[1].set_title('Weekly Plot')
axes[1].set_xlabel('Weekly')
axes[1].set_ylabel('')

axes[2].set_title('Yearly Plot')
axes[2].set_xlabel('Month')
axes[2].set_ylabel('')

st.subheader('Trend Components')
subheader = 'First chart refers to the trend of the stock.'
st.markdown(f"<p style='font-family: sans-serif; font-size: 14px; font-style: italic;'>{subheader}</p>", unsafe_allow_html=True)
subheader = 'Second chart refers to the weekly trend of the stock.'
st.markdown(f"<p style='font-family: sans-serif; font-size: 14px; font-style: italic;'>{subheader}</p>", unsafe_allow_html=True)
subheader = 'Third chart refers to the annual tenure of the stock.'
st.markdown(f"<p style='font-family: sans-serif; font-size: 14px; font-style: italic;'>{subheader}</p>", unsafe_allow_html=True)
st.write(fig2)

#######################################################################################################

fig3 = m.plot(forecast)
axes = fig3.get_axes()
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Price')

a = add_changepoints_to_plot(fig3.gca(), m, forecast)

st.write(" ")
st.subheader('ChangePoints Plot')
subheader = 'The <strong>Changepoints</strong> are the date points at which the time series exhibit abrupt changes in trajectory.'
st.markdown(f"<p style='font-family: sans-serif; font-size: 14px; font-style: italic;'>{subheader}</p>", unsafe_allow_html=True)
st.write(fig3)


