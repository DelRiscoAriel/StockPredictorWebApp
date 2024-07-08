# pip install streamlit yfinance prophet plotly
import streamlit as st 
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

#Set dates
START = '2015-01-01'
TODAY = date.today().strftime('%Y-%m-%d')

st.title("Stock Predictor App")

ticker = st.text_input("Ticker name: ", "AAPL")

period = st.slider("Days of predictions", 7, 365)

#Lead Stock Data from the online API
@st.cache_data #Save data in cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(ticker)
data_load_state = st.text("Loading data... done")

st.subheader('Raw Data')
st.write(data.tail()) #Display last 10 rows

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Predictor
features = ['Date', 'Close']
df_train = data[features]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
predictions = model.predict(future)

st.subheader('Forecast Data')
st.write(predictions.tail())

st.write('Forecast Data')
fig1 = plot_plotly(model, predictions)
st.plotly_chart(fig1)

st.write('Forecast Components')
fig2 = model.plot_components(predictions)
st.write(fig2)
