import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from statsmodels.tsa.arima.model import ARIMA

# Load saved ARIMA model from file using pickle
with open('arima_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define function to make forecasts using the loaded model
def make_forecasts(model, data, steps):
    # Make forecasts using loaded model
    forecast = model.forecast(steps=steps)

    # Return forecasts as a Pandas Series
    return pd.Series(forecast, index=pd.date_range(start=data.index[-1], periods=steps+1, freq='M')[1:])

# Set up Streamlit app
st.title('Demand Forecaster(ARIMA)')

# Allow user to input forecasting horizon
steps = st.slider('Forecasting horizon (in months)', min_value=1, max_value=24, value=12)

# Load data (or generate dummy data for demonstration purposes)
data = pd.read_csv('monthly_data.csv', index_col=0, parse_dates=True)

# Make forecasts using the loaded model
forecast = make_forecasts(model, data, steps)

# Display forecasts in a table
st.write('Forecasted values:')
st.write(forecast)

# Plot time series data and forecasts
fig, ax = plt.subplots()
data.plot(ax=ax)
forecast.plot(ax=ax, label='Forecast')
ax.set(title='Demand forecast', xlabel='Month', ylabel='Demand')
ax.legend()
st.pyplot(fig)
