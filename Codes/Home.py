#cd Desktop
#pip install --upgrade protobuf --user
#streamlit run c:\Users\user\Desktop\Codes\Home.py

import streamlit as st
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

import pandas as pd
import fundamentalanalysis as fa


import pickle
from pathlib import Path 

import streamlit as st 
import streamlit_authenticator as stauth 

st.set_page_config(page_title="Home", layout="wide")

#####################################################################################

names = ["Loo Yeen Chenng", "Jonavhan Tan Quaan Vmin","Kong Yan Yi", "Nicholas Low Jia Jun", "Foo Jin Wei"] 
usernames = ["Eve", "Whiskybonbon", "Kong Yan Yi", "niicholaslow", "Benzi"] 

file_path = Path('c:/Users/user/Desktop/Codes') / "credentials.pkl"
with file_path.open("rb") as file: 
    hashed_passwords = pickle.load(file) 

authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "prediction", "abcdef", cookie_expiry_days=0.017)

name, authentication_status, username = authenticator.login('Login', 'main') 

#####################################################################################

# Set default theme to light
def set_theme_light():
    st.markdown(
        """
        <style>
        body {
            color: black;
            background-color: white;
        }
        .reportview-container .main .block-container {
            max-width: 1000px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Set default theme to light
set_theme_light()

if st.session_state["authentication_status"]:

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

    @ st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data = load_data(selected_stock)

    authenticator.logout('Logout', 'main')

#################################################################################################################
#################################################################################################################

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
################################################################################################################################

    @st.cache_resource
    def profile(ticker, api_key):
        df_summary = fa.profile(ticker, api_key)
        return df_summary
    df_summary = profile(selected_stock, api_key)

    @ st.cache_data()
    def entreprise_value(ticker, api_key):
        df_value = fa.enterprise(ticker, api_key)
        return df_value
    df_value = entreprise_value(selected_stock, api_key)
    df_value = df_value[['2022']]

    @st.cache_resource
    def financial_ratios_annually(ticker, api_key, period="annual"):
        df_ratios = fa.financial_ratios(ticker, api_key, period="annual")
        return df_ratios
    df_ratios = financial_ratios_annually(selected_stock, api_key, period="annual")
    df_ratios = df_ratios[['2022']]

    @st.cache_data()
    def dividends(ticker, api_key):
        df_dividends = fa.stock_dividend(ticker, api_key)
        return df_dividends
    df_dividends = dividends(selected_stock, api_key)

    @st.cache_data()
    def quotes(ticker, api_key):
        df_quote = fa.quote(ticker, api_key)
        return df_quote
    df_quote = quotes(selected_stock, api_key)

    @st.cache_data()
    def ratings(ticker, api_key):
        df_rating = fa.rating(ticker, api_key)
        return df_rating
    df_rating = ratings(selected_stock, api_key)


    DescStats = [
                {'Label': 'Price (USD)', 'Value': df_summary[0][1]},
                {'Label': 'Percentage Change', 'Value': df_quote[0][3]},
                {'Label': 'Open', 'Value': df_quote[0][15]},
                {'Label': 'Close', 'Value': df_quote[0][16]},
                {'Label': '52-Week Range', 'Value': '$' + str(df_summary[0][6]).replace('-', ' - $')},
        ]
    st.markdown('### Metrics')
    col1, col2, col3, col4 = st.columns(4)

    # Modify the labels and values accordingly
    rounded_value1 = round(float(df_summary[0][1]), 2)
    rounded_change1 = round(float(df_quote[0][3]), 2)
    rounded_value2 = round(float(df_quote[0][15]), 2)
    rounded_value3 = round(float(df_quote[0][16]), 2)
    rounded_value4 = (df_summary[0][6])


    label1 = "Price (USD)"
    value1 = rounded_value1
    change1 = rounded_change1

    label2 = "Open (USD)"
    value2 = rounded_value2

    label3 = "Close (USD)"
    value3 = rounded_value3

    label4 = "52-Week Range (USD)"
    value4 = rounded_value4


    formatted_value1 = format(value1)
    col1.metric(label1, formatted_value1, f"{change1}%")

    formatted_value2 = "{:,}".format(value2)
    col2.metric(label2, formatted_value2)

    formatted_value3 = "{:,}".format(value3)
    col3.metric(label3, formatted_value3)

    formatted_value4 = (value4)
    col4.metric(label4, formatted_value4)


################################################################################################################


    def plot_candle_data():
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data['Date'],
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Market Movement'))
        fig.update_layout(
            # title='Share Price Movement',
            yaxis_title='USD per Share',
            yaxis=dict(tickprefix='$'),
            showlegend=False,
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)

    # Add subheader
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.subheader('Share Price Movement')
    subheader = "Visually represents the stock's open, high, low, and close price for each day. Provides you valuable information about price trends, market sentiment, and the range between the day's high and low."
    st.markdown(f"<p style='font-family: sans-serif; font-size: 14px; font-style: italic;'>{subheader}</p>", unsafe_allow_html=True)
    plot_candle_data()



    # #Returns
    # data_returns = data[['Date', 'Adj Close']]
    # data_returns.rename(columns={'Adj Close' : 'price_t'}, inplace = True)
    # data_returns['returns'] = (data_returns['price_t'] / data_returns['price_t'].shift(1)) - 1

    # def plot_returns_data():
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x = data_returns['Date'], y = data_returns['returns'], hovertemplate='Returns: %{y:.2%}<extra></extra>'))
    #     fig.layout.update(xaxis_rangeslider_visible = True,
    #     height = 600, yaxis_title='Returns Percentage %')
    #     st.plotly_chart(fig, use_container_width=True)


    # plot_returns_data()


    data.reset_index(inplace=True)
    data.set_index('Date', inplace=True)


    filtered_data = data.loc[START:TODAY]
    def plot_ma_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Close'], name='Close', line=dict(color='black'), hovertemplate='Close: $%{y:.2f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Close'].rolling(window=50).mean(), name='50-day MA', line=dict(color='red'), hovertemplate='50-day MA: $%{y:.2f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Close'].rolling(window=200).mean(), name='200-day MA', line=dict(color='blue'), hovertemplate='200-day MA: $%{y:.2f}<extra></extra>'))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)
    
    # Add subheader
    st.write(" ")
    st.subheader('Moving Averages')
    subheader = "It overlays the closing price of the stock with moving averages of different time windows. You can identify potential buy or sell signals."
    st.markdown(f"<p style='font-family: sans-serif; font-size: 14px; font-style: italic;'>{subheader}</p>", unsafe_allow_html=True)
    plot_ma_data()


    def plot_volatility_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'].pct_change().rolling(window=21).std(), name='Volatility', hovertemplate='Date: %{x}<br>Volatility: %{y:.2%}<extra></extra>'))
        fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Volatility',
                    xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Add subheader
    st.write(" ")
    st.subheader('Volatility - Rolling standard deviation of the stock daily percentage change')
    subheader = "It provides insights into the stock's price volatility, indicating periods of higher or lower volatility. You can use this information to assess the level of risk associated with the stock and adjust your investment strategies accordingly."
    st.markdown(f"<p style='font-family: sans-serif; font-size: 14px; font-style: italic;'>{subheader}</p>", unsafe_allow_html=True)
    plot_volatility_data()


    import calendar

    def plot_yearlyavg_data():
        # Get the unique years in the data
        years = data.index.year.unique()

        # Create a selectbox to filter by year
        selected_year = st.selectbox('Select Year', years)

        # Filter the DataFrame for the selected year
        filtered_data = data[data.index.year == selected_year]

        # Calculate monthly statistics: min, max, and average closing price for each month
        monthly_stats = filtered_data.groupby(filtered_data.index.month)['Close'].agg(['min', 'max', 'mean'])

        fig = go.Figure()

        # Iterate over each month
        for month in range(1, 13):
            # Check if the month exists in the monthly_stats DataFrame
            if month in monthly_stats.index:
                # Filter data for the specific month
                monthly_data = monthly_stats.loc[month]

                # Create a box plot for the month
                fig.add_trace(go.Box(
                    y=monthly_data,
                    name=calendar.month_name[month],
                    boxpoints='all',
                    jitter=0.5,
                    whiskerwidth=0.2,
                    marker=dict(size=2),
                    line=dict(width=1),
                    hovertemplate='Value: $%{y:.2f}<extra></extra>'
                ))

        fig.update_layout(
            title='Yearly Closing Price Statistics',
            xaxis_title='Month',
            yaxis_title='Price'
        )

        # Render the figure using Streamlit
        st.plotly_chart(fig, use_container_width=True)

    # ##########Default has no value
    # def plot_yearlyavg_data():
    #     # Get the unique years in the data
    #     years = data.index.year.unique()

    #     # Create a multiselect to filter by year
    #     selected_years = st.multiselect('Select Years', years)

    #     # Filter the DataFrame for the selected years
    #     filtered_data = data[data.index.year.isin(selected_years)]

    #     # Calculate monthly statistics: min, max, and average closing price for each month
    #     monthly_stats = filtered_data.groupby(filtered_data.index.month)['Close'].agg(['min', 'max', 'mean'])

    #     fig = go.Figure()

    #     # Iterate over each month
    #     for month in range(1, 13):
    #         # Check if the month exists in the monthly_stats DataFrame
    #         if month in monthly_stats.index:
    #             # Filter data for the specific month
    #             monthly_data = monthly_stats.loc[month]

    #             # Create a box plot for the month
    #             fig.add_trace(go.Box(
    #                 y=monthly_data,
    #                 name=calendar.month_name[month],
    #                 boxpoints='all',
    #                 jitter=0.5,
    #                 whiskerwidth=0.2,
    #                 marker=dict(size=2),
    #                 line=dict(width=1),
    #                 hovertemplate='Value: $%{y:.2f}<extra></extra>'
    #             ))

    #     fig.update_layout(
    #         title='Seasonal Analysis - Monthly Closing Price Statistics',
    #         xaxis_title='Month',
    #         yaxis_title='Price'
    #     )

    #     # Render the figure using Streamlit
    #     st.plotly_chart(fig, use_container_width=True)

    # plot_yearlyavg_data()


    ###########Default select all year
    def plot_yearlyavg_data2():
        # Get the unique years in the data
        years = data.index.year.unique()

        # Create a multiselect to filter by year, with all years selected by default
        selected_years = st.multiselect('Select Years', options=list(years), default=list(years))

        # Filter the DataFrame for the selected years
        filtered_data = data[data.index.year.isin(selected_years)]

        # Calculate monthly statistics: min, max, and average closing price for each month
        monthly_stats = filtered_data.groupby(filtered_data.index.month)['Close'].agg(['min', 'max', 'mean'])

        fig = go.Figure()

        # Iterate over each month
        for month in range(1, 13):
            # Check if the month exists in the monthly_stats DataFrame
            if month in monthly_stats.index:
                # Filter data for the specific month
                monthly_data = monthly_stats.loc[month]

                # Create a box plot for the month
                fig.add_trace(go.Box(
                    y=monthly_data,
                    name=calendar.month_name[month],
                    boxpoints='all',
                    jitter=0.5,
                    whiskerwidth=0.2,
                    marker=dict(size=2),
                    line=dict(width=1),
                    hovertemplate='Value: $%{y:.2f}<extra></extra>'
                ))

        fig.update_layout(
            title='Cumulative Yearly Closing Price Statistics',
            xaxis_title='Month',
            yaxis_title='Price'
        )

        # Render the figure using Streamlit
        st.plotly_chart(fig, use_container_width=True)
    
    # Add subheader
    st.write(" ")
    st.subheader('Seasonal Analysis')
    subheader = "You can identify any recurring patterns or trends in the stock's price movement over different years as well as understanding seasonal influences on the stock's performance. You also able to compare between years to identify changes or shifts in the stock's seasonal behaviour over time."
    st.markdown(f"<p style='font-family: sans-serif; font-size: 14px; font-style: italic;'>{subheader}</p>", unsafe_allow_html=True)
    plot_yearlyavg_data()
    plot_yearlyavg_data2()



    data['Return'] = data['Close'].pct_change()

    # def plot_distribution_data():
    #     fig = go.Figure()
    #     fig.add_trace(
    #         go.Histogram(
    #             x=data['Return'],
    #             marker=dict(
    #                 color='#3498db',  # Use a modern blue color
    #                 line=dict(
    #                     color='black',
    #                     width=1
    #                 )
    #             ),
    #             hoverinfo='x+y',
    #             hovertemplate="Return: %{x:.2%} <br>Count: %{y}",
    #             name='Return Distribution'
    #         )
    #     )
    #     fig.update_layout(
    #         title='Price Change Distribution',
    #         xaxis_title='Return',
    #         yaxis_title='Frequency',
    #         bargap=0.1,
    #         hovermode='x unified',
    #         plot_bgcolor='white',  # Set the plot background color to white
    #         paper_bgcolor='white',  # Set the paper background color to white
    #         font=dict(color='black')  # Set the font color to black
    #     )
    #     st.plotly_chart(fig, use_container_width=True)

    # plot_distribution_data()


    def plot_pairplot_data():
        # Define subplot structure
        fig = make_subplots(rows=2, cols=2, 
                            subplot_titles=('<b>Price vs Volume</b>', '<b>Price vs Return</b>', '<b>Volume vs Return</b>'),
                            vertical_spacing=0.12, 
                            horizontal_spacing=0.12)

        # Price vs Volume
        fig.add_trace(
            go.Scattergl(x=data['Close'], y=data['Volume'], mode='markers', 
                        marker=dict(color='rgb(0, 0, 255)', 
                                    line=dict(width=1, color='DarkSlateGrey')),
                        hovertemplate='Price: $%{x}<br>Volume: %{y}'),
            row=1, col=1
        )
        fig.update_xaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=1, col=1)

        # Price vs Return
        fig.add_trace(
            go.Scattergl(x=data['Close'], y=data['Return'], mode='markers', 
                        marker=dict(color='rgb(127, 127, 127)', 
                                    line=dict(width=1)),
                        hovertemplate='Price: $%{x}<br>Return: %{y:.2%}'),
            row=1, col=2
        )
        fig.update_xaxes(title_text='Price', row=1, col=2)
        fig.update_yaxes(title_text='Return', row=1, col=2)

        # Volume vs Return
        fig.add_trace(
            go.Scattergl(x=data['Volume'], y=data['Return'], mode='markers', 
                        marker=dict(color='rgb(255, 180, 0)', 
                                    line=dict(width=1)),
                        hovertemplate='Volume: %{x}<br>Return: %{y:.2%}'),
            row=2, col=1
        )
        fig.update_xaxes(title_text='Volume', row=2, col=1)
        fig.update_yaxes(title_text='Return', row=2, col=1)

        # Update layout
        fig.update_layout(height=800, width=800, title_text="Pairplot")
        st.plotly_chart(fig, use_container_width=True)

    plot_pairplot_data()

################################################################################################################

    st.markdown('### Additional Metrics')
    AddDescStats = [
    {'Label': 'Symbol', 'Value': str(df_summary[0][0])},
    {'Label': 'Price', 'Value': '${:.2f}'.format(float(df_summary[0][1]))},
    {'Label': 'Beta', 'Value': '{:.2f}'.format(float(df_summary[0][2]))},
    {'Label': 'Volume Average', 'Value': '${:,.0f}'.format(float(df_summary[0][3]))},
    {'Label': 'Market Capitalisation', 'Value': '${:,.0f}'.format(float(df_summary[0][4]))},
    {'Label': 'Last Dividend', 'Value': '{:.2f}%'.format(float(df_summary[0][5]))},
    {'Label': 'Range', 'Value': '$' + str(df_summary[0][6]).replace('-', ' - $')},
    {'Label': 'Company Name', 'Value': str(df_summary[0][8])},
    {'Label': 'Industry', 'Value': str(df_summary[0][15])},
    {'Label': 'CEO', 'Value': str(df_summary[0][18])},
    {'Label': 'Sector', 'Value': str(df_summary[0][19])},
    {'Label': 'Website', 'Value': str(df_summary[0][16])},
    {'Label': 'Address', 'Value': str(df_summary[0][23])},
    {'Label': 'IPO Date', 'Value': str(df_summary[0][30])},
    {'Label': 'Full-Time Employees', 'Value': '{:,.0f}'.format(float(df_summary[0][21]))},
]

    col1 = st.columns(1)

    with col1[0].expander("Stock Details", expanded=True):
        for item in AddDescStats:
            st.markdown(f"<div style='display:flex; justify-content:space-between; margin-bottom:10px;'> <span style='text-align:left;'>{item['Label']}</span> <span style='text-align:right; font-weight:bold; font-family:sans-serif; font-stretch: condensed; font-size:12px;'>{item['Value']}</span> </div>", unsafe_allow_html=True)
   
################################################################################################################


################################################################################################################

elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')

    def add_logo(logo_path, width, height):
        logo = Image.open("c:/Users/user/Desktop/Codes/Logo(2).png")
        modified_logo = logo.resize((width, height))
        return modified_logo
    my_logo = add_logo(logo_path="c:/Users/user/Desktop/Codes/Logo(2).png", width=300, height=300)
    st.sidebar.image(my_logo)
    
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
            <h3><strong>WELCOME TO BULL AND BEAR</strong></h3>
        </div>
    """, unsafe_allow_html=True)
    
elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')

    def add_logo(logo_path, width, height):
        logo = Image.open("c:/Users/user/Desktop/Codes/Logo(2).png")
        modified_logo = logo.resize((width, height))
        return modified_logo
    my_logo = add_logo(logo_path="c:/Users/user/Desktop/Codes/Logo(2).png", width=300, height=300)
    st.sidebar.image(my_logo)

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
            <h3><strong>WELCOME TO BULL AND BEAR</strong></h3>
        </div>
    """, unsafe_allow_html=True)

