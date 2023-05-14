import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
import pickle

# Define the Streamlit app
def app():
    # Set the page title and heading
    st.set_page_config(page_title='Exponential Smoothing Forecast')
    st.title('Exponential Smoothing Forecast')

    # Create a file uploader widget
    uploaded_file = st.file_uploader('Upload a CSV file containing y_test')

    # If a file was uploaded, read it into a Pandas dataframe
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)

        train = data.loc[data.index < '2021-03-31']
        test = data.loc[data.index >= '2021-03-31']
        
        # Load the Exponential Smoothing model from a pickle file
        with open('ES_model.pkl', 'rb') as f:
            model_ES = pickle.load(f)

        # Generate forecasts for the testing set
        y_pred_ses = model_ES.forecast(len(test))

        # Plot the actual values and forecast
        plt.figure(figsize=(12,6))
        plt.plot(train.index, train.values, label='Actual Train')
        plt.plot(test.index, test.values, label='Actual Test')
        plt.plot(test.index, y_pred_ses, label='Forecast')
        plt.legend()
        plt.title('Exponential Smoothing Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        st.pyplot(plt)

# Run the Streamlit app
if __name__ == '__main__':
    app()
