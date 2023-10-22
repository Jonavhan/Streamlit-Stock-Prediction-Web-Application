import streamlit as st
import yfinance as yf
from datetime import date
import fundamentalanalysis as fa
import pandas as pd

import streamlit as st


st.write("")
st.write("")
st.title("Stock Information")


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

START = "2012-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

selected_stock=st.session_state['selected_stock']
n_years = st.session_state['n_years']
period = st.session_state['period']

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

tab_titles = [
    "Company Profile",
    "Enterprise Value",
    "Financial Ratios",
    "Dividend",
    "Financial Statements",
]

tabs = st.tabs(tab_titles)

#----------------------Company Profile Tab--------------------------------------------------------------------------------------------------------------------------------

#CompanyProfile
with tabs[0]:
    
    @st.cache_data
    def profile(ticker, api_key):
        df_summary = fa.profile(ticker, api_key)
        return df_summary

    df_summary = profile(selected_stock, api_key)


#----------------------1st Dropdown--------------------------------------------------------------------------------------------------------------------------------
    StockDescription = [
        {'Label': '', 'Value': df_summary[0][17]},
]
    col1 = st.columns(1)

    with col1[0].expander("Company Background", expanded=True):
        for item in StockDescription:
            st.markdown(f"<div style='display:flex; justify-content:space-between; text-align:justify; margin-bottom:10px;'> <span style='text-align:justify;'>{item['Label']}</span> <span style='text-align:justify; font-family:sans-serif; font-size:14px;'>{item['Value']}</span> </div>", unsafe_allow_html=True)
    
#----------------------Dataframe--------------------------------------------------------------------------------------------------------------------------------

    CompanyProfileDF = [
    {'Label': 'Symbol', 'Value': str(df_summary[0][0])},
    {'Label': 'Price', 'Value': '${:.2f}'.format(float(df_summary[0][1]))},
    {'Label': 'Beta', 'Value': '{:.2f}'.format(float(df_summary[0][2]))},
    {'Label': 'Volume Average', 'Value': '${:,.0f}'.format(float(df_summary[0][3]))},
    {'Label': 'Market Capitalisation', 'Value': '${:,.0f}'.format(float(df_summary[0][4]))},
    {'Label': 'Last Dividend', 'Value': '${:.2f}'.format(float(df_summary[0][5]))},
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
        for item in CompanyProfileDF:
            st.markdown(f"<div style='display:flex; justify-content:space-between; margin-bottom:10px;'> <span style='text-align:left;'>{item['Label']}</span> <span style='text-align:right; font-weight:bold; font-family:sans-serif; font-stretch: condensed; font-size:12px;'>{item['Value']}</span> </div>", unsafe_allow_html=True)



#-------------------------------------------------------------------------------------------------------------------------------------------------------------

#EnterpriseValue
with tabs[1]:
    st.markdown("<p style='font-family: sans-serif; font-size: 14px; font-style: italic;'>Please note that all the data presented reflects the most recent updates available as of the year 2022.</p>", unsafe_allow_html=True)

    @st.cache_data()
    def entreprise_value(ticker, api_key):
        df_value = fa.enterprise(ticker, api_key)
        return df_value

    df_value = entreprise_value(selected_stock, api_key)
    df_value = df_value[['2022']]

    EnterpriseValueDF = [
    {'Label': 'Number of Shares', 'Value': "${:,.0f}".format(df_value['2022'][2])},
    {'Label': 'Minus Cash & Cash Equivalents', 'Value': "${:,.0f}".format(df_value['2022'][4])},
    {'Label': 'Add Total Debt', 'Value': "${:,.0f}".format(df_value['2022'][5])},
    {'Label': 'Enterprise Value', 'Value': "${:,.0f}".format(df_value['2022'][6])},
]

    col1 = st.columns(1)

    with col1[0].expander("Enterprise Values", expanded=True):
        for item in EnterpriseValueDF:
            st.markdown(f"<div style='display:flex; justify-content:space-between; margin-bottom:10px;'> <span style='text-align:left;'>{item['Label']}</span> <span style='text-align:right; font-weight:bold; font-family:sans-serif; font-stretch: condensed; font-size:12px;'>{item['Value']}</span> </div>", unsafe_allow_html=True)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------

#FinancialRatios
with tabs[2]:
    st.markdown("<p style='font-family: sans-serif; font-size: 14px; font-style: italic;'>Please note that all the data presented reflects the most recent updates available as of the year 2022.</p>", unsafe_allow_html=True)

    @st.cache_data
    def financial_ratios_annually(ticker, api_key, period="annual"):
        df_ratios = fa.financial_ratios(ticker, api_key, period="annual")
        return df_ratios
    
    df_ratios = financial_ratios_annually(selected_stock, api_key, period="annual")
    df_ratios = df_ratios[['2022']]

    FinancialRatiosDF = [
    {'Label': 'Current Ratio', 'Value': "{:,.5f}".format(df_ratios['2022'][1])},
    {'Label': 'Quick Ratio', 'Value': "{:,.5f}".format(df_ratios['2022'][2])},
    {'Label': 'Cash Ratio', 'Value': "{:,.5f}".format(df_ratios['2022'][3])},
    {'Label': 'Gross Profit Margin', 'Value': "{:,.5f}".format(df_ratios['2022'][9])},
    {'Label': 'Operating Profit Margin', 'Value': "{:,.5f}".format(df_ratios['2022'][10])},
    {'Label': 'Net Profit Margin', 'Value': "{:,.5f}".format(df_ratios['2022'][11])},
    {'Label': 'Return on Assets (ROA)', 'Value': "{:,.5f}".format(df_ratios['2022'][14])},
    {'Label': 'Return on Equity (ROE)', 'Value': "{:,.5f}".format(df_ratios['2022'][15])},
    {'Label': 'Debt Ratio', 'Value': "{:,.5f}".format(df_ratios['2022'][20])},
    {'Label': 'Interest Coverage', 'Value': "{:,.5f}".format(df_ratios['2022'][24])},
]

    col1 = st.columns(1)

    with col1[0].expander("Enterprise Values", expanded=True):
        for item in FinancialRatiosDF:
            st.markdown(f"<div style='display:flex; justify-content:space-between; margin-bottom:10px;'> <span style='text-align:left;'>{item['Label']}</span> <span style='text-align:right; font-weight:bold; font-family:sans-serif; font-stretch: condensed; font-size:12px;'>{item['Value']}</span> </div>", unsafe_allow_html=True)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Dividend
with tabs[3]:
    st.markdown("Display complete information about the company's dividend which includes adjusted dividend, dividend, record date, payment date and declaration date over time.")

    @st.cache_data()
    def dividends(ticker, api_key):
        df_dividends = fa.stock_dividend(ticker, api_key)
        return df_dividends
    
    df_dividends = dividends(selected_stock, api_key)

    DividendDF = [
    {'Label': df_dividends['declarationDate'][0], 'Value': str(df_dividends['dividend'][0]) + '%'},
    {'Label': df_dividends['declarationDate'][1], 'Value': str(df_dividends['dividend'][1]) + '%'},
    {'Label': df_dividends['declarationDate'][2], 'Value': str(df_dividends['dividend'][2]) + '%'},
    {'Label': df_dividends['declarationDate'][3], 'Value': str(df_dividends['dividend'][3]) + '%'},
    {'Label': df_dividends['declarationDate'][4], 'Value': str(df_dividends['dividend'][4]) + '%'},
    {'Label': df_dividends['declarationDate'][5], 'Value': str(df_dividends['dividend'][5]) + '%'},
    {'Label': df_dividends['declarationDate'][6], 'Value': str(df_dividends['dividend'][6]) + '%'},
    {'Label': df_dividends['declarationDate'][7], 'Value': str(df_dividends['dividend'][7]) + '%'},
    {'Label': df_dividends['declarationDate'][8], 'Value': str(df_dividends['dividend'][8]) + '%'},
    {'Label': df_dividends['declarationDate'][9], 'Value': str(df_dividends['dividend'][9]) + '%'},
    {'Label': df_dividends['declarationDate'][10], 'Value': str(df_dividends['dividend'][10]) + '%'},
]


    header_left = 'Declaration Date'
    header_right = 'Dividend'

    col1 = st.columns(1)

    with col1[0].expander("Dividend Details", expanded=True):
        st.markdown(f"<div style='display:flex; justify-content:space-between; margin-bottom:10px;'> <span style='text-align:left; font-weight:bold;'>{header_left}</span> <span style='text-align:right; font-weight:bold;'>{header_right}</span> </div>", unsafe_allow_html=True)
        for item in DividendDF:
            st.markdown(f"<div style='display:flex; justify-content:space-between; margin-bottom:10px;'> <span style='text-align:left;'>{item['Label']}</span> <span style='text-align:right;'>{item['Value']}</span> </div>", unsafe_allow_html=True)






#------------------------------------------------------------------------------------------------------------------------------------------------------------------

#IncomeStatement
with tabs[4]:
    st.markdown("<p style='font-family: sans-serif; font-size: 14px; font-style: italic;'>Please note that all the data presented reflects the most recent updates available as of the year 2022.</p>", unsafe_allow_html=True)

    @st.cache_data()
    def income_statement (ticker, api_key):
        df_income_statement = fa.income_statement (ticker, api_key)
        return df_income_statement
    
    df_income_statement = income_statement (selected_stock, api_key)

    IncomeStatementDF = [
    {'Label': 'Revenue', 'Value': '${:,.0f}'.format(df_income_statement['2022'][6])},
    {'Label': 'Gross Profit', 'Value': '${:,.0f}'.format(df_income_statement['2022'][8])},
    {'Label': 'Gross Profit Ratio', 'Value': format(df_income_statement['2022'][9], ',.4f')},
    {'Label': 'Operating Income', 'Value': '${:,.0f}'.format(df_income_statement['2022'][22])},
    {'Label': 'Operating Income Ratio', 'Value': format(df_income_statement['2022'][23], ',.4f')},
    {'Label': 'Net Income', 'Value': '${:,.0f}'.format(df_income_statement['2022'][28])},
    {'Label': 'Net Income Ratio', 'Value': format(df_income_statement['2022'][29], ',.4f')},
    {'Label': 'Earnings Per Share (EPS)', 'Value': format(df_income_statement['2022'][30], ',.4f')},
]

    col1 = st.columns(1)

    with col1[0].expander("Income Statement", expanded=False):
        for item in IncomeStatementDF:
            st.markdown(f"<div style='display:flex; justify-content:space-between; margin-bottom:10px;'> <span style='text-align:left;'>{item['Label']}</span> <span style='text-align:right; font-weight:bold; font-family:sans-serif; font-stretch: condensed; font-size:12px;'>{item['Value']}</span> </div>", unsafe_allow_html=True)


#BalanceSheetStatement

    @st.cache_data()
    def balance_sheet_statement (ticker, api_key):
        df_balance_ss = fa.balance_sheet_statement (ticker, api_key)
        return df_balance_ss
    
    df_balance_ss = balance_sheet_statement (selected_stock, api_key)

    BalanceSheetDF = [
    {'Label': 'Cash and Cash Equivalents', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][6])},
    {'Label': 'Short-Term Investments', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][7])},
    {'Label': 'Net Receivables', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][9])},
    {'Label': 'Inventory', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][10])},
    {'Label': 'Total Current Assets', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][12])},
    {'Label': 'Property Plant Equipment Net', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][13])},
    {'Label': 'Total Non-Current Assets', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][20])},
    {'Label': 'Total Assets', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][22])},
    {'Label': 'Account Payables', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][23])},
    {'Label': 'Short-Term Debt', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][24])},
    {'Label': 'Total Current Liabilities', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][28])},
    {'Label': 'Long-Term Debt', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][29])},
    {'Label': 'Total Non-Current Liabilities', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][33])},
    {'Label': 'Total Liabilities', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][36])},
    {'Label': 'Total Stockholders Equity', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][42])},
    {'Label': 'Total Equity', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][43])},
    {'Label': 'Total Liabilities & Stockholders Equity', 'Value': '${:,.0f}'.format(df_balance_ss['2022'][44])},
]


    col1 = st.columns(1)

    with col1[0].expander("Balance Sheet", expanded=False):
        for item in BalanceSheetDF:
            st.markdown(f"<div style='display:flex; justify-content:space-between; margin-bottom:10px;'> <span style='text-align:left;'>{item['Label']}</span> <span style='text-align:right; font-weight:bold; font-family:sans-serif; font-stretch: condensed; font-size:12px;'>{item['Value']}</span> </div>", unsafe_allow_html=True)



#CashFlowStatement

    @st.cache_data()
    def cash_flow_statement (ticker, api_key):
        df_cfs = fa.cash_flow_statement (ticker, api_key)
        return df_cfs
    
    df_cfs = cash_flow_statement (selected_stock, api_key)

    CashFlowDF = [
    {'Label': 'Net Cash Provided by Operating Activities', 'Value': '${:,.0f}'.format(df_cfs['2022'][16])},
    {'Label': 'Net Cash Used For Investing Activities', 'Value': '-${:,.0f}'.format(abs(df_cfs['2022'][22]))},
    {'Label': 'Net Cash Used Provided By Financing Activities', 'Value': '-${:,.0f}'.format(abs(df_cfs['2022'][28]))},
    {'Label': 'Free Cash Flow', 'Value': '${:,.0f}'.format(df_cfs['2022'][35])},
    {'Label': 'Net Change In Cash', 'Value': '${:,.0f}'.format(df_cfs['2022'][30])},
    {'Label': 'Cash At End Of Period', 'Value': '${:,.0f}'.format(df_cfs['2022'][31])},
    {'Label': 'Cash At Beginning Of Period', 'Value': '${:,.0f}'.format(df_cfs['2022'][32])},
    {'Label': 'Operating Cash Flow', 'Value': '${:,.0f}'.format(df_cfs['2022'][33])},
    {'Label': 'Capital Expenditure', 'Value': '-${:,.0f}'.format(abs(df_cfs['2022'][34]))},
]

    col1 = st.columns(1)

    with col1[0].expander("Cash Flow", expanded=False):
        for item in CashFlowDF:
            st.markdown(f"<div style='display:flex; justify-content:space-between; margin-bottom:10px;'> <span style='text-align:left;'>{item['Label']}</span> <span style='text-align:right; font-weight:bold; font-family:sans-serif; font-stretch: condensed; font-size:12px;'>{item['Value']}</span> </div>", unsafe_allow_html=True)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
