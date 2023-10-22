# import openai
# import streamlit as st
# from streamlit_pills import pills

import streamlit as st

st.write(" ")
st.write(" ")
st.title("AI Assistant")


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



# openai.api_key = "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"

# st.subheader("AI Assistant : Streamlit + OpenAI: `stream` *argument*")
# selected = pills("", ["NO Streaming", "Streaming"])

# user_input = st.text_input("You: ",placeholder = "Ask me anything ...", key="input")


# if st.button("Submit", type="primary"):
#     st.markdown("----")
#     res_box = st.empty()
#     if selected == "Streaming":
#         report = []
#         for resp in openai.Completion.create(model='gpt-3.5-turbo',
#                                             prompt=user_input,
#                                             max_tokens=500, 
#                                             temperature = 0.5,
#                                             stream = True):
#             # join method to concatenate the elements of the list 
#             # into a single string, 
#             # then strip out any empty strings
#             report.append(resp.choices[0].text)
#             result = "".join(report).strip()
#             result = result.replace("\n", "")        
#             res_box.markdown(f'*{result}*') 
            
#     else:
#         completions = openai.Completion.create(model='gpt-3.5-turbo',
#                                             prompt=user_input,
#                                             max_tokens=500, 
#                                             temperature = 0.5,
#                                             stream = False)
#         result = completions.choices[0].text
        
#         res_box.write(result)
# st.markdown("----")


import openai
import streamlit as st
from streamlit_pills import pills

# Set up OpenAI API credentials (consider using environment variables)
openai.api_key = "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"

# Streamlit layout
st.markdown("**No streaming** is the API call will block until the model generates a complete response. This means that the response is returned only after the model has finished generating the entire output text. It is useful when you want to get the complete response from the model before processing it further.")
st.markdown("**Streaming** is the API call returns a generator object that allows you to retrieve the response in chunks as it is being generated by the model. This is useful when you want to process the response in real-time or handle large outputs in a more memory-efficient manner.")
selected = pills("", ["NO Streaming", "Streaming"])

user_input = st.text_input("You:", placeholder="Ask me anything...", key="input")

if st.button("Submit", type="primary"):
    st.markdown("----")
    res_box = st.empty()

    if not selected:
        st.warning("Please select a streaming option.")
    elif not user_input:
        st.warning("Please provide a user input.")
    else:
        try:
            if selected == "Streaming":
                report = []
                for resp in openai.Completion.create(
                    model='text-davinci-003',
                    prompt=user_input,
                    max_tokens=500,
                    temperature=0.5,
                    stream=True
                ):
                    report.append(resp.choices[0].text)
                    result = "".join(report).strip().replace("\n", "")
                    res_box.markdown(f'*{result}*')
            else:
                completions = openai.Completion.create(
                    model='text-davinci-003',
                    prompt=user_input,
                    max_tokens=500,
                    temperature=0.5,
                    stream=False
                )
                result = completions.choices[0].text
                res_box.write(result)
        except Exception as e:
            st.error("An error occurred. Please try again.")

st.markdown("----")


