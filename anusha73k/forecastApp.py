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
        y_test = pd.read_csv(uploaded_file, index_col=0, parse_dates=True, squeeze=True)

        # Load the Exponential Smoothing model from a pickle file
        with open('anusha73k\ES_model.pkl', 'rb') as f:
            model_ES = pickle.load(f)

        # Generate forecasts for the testing set
        y_pred_ses = model_ES.forecast(len(y_test))

        # Plot the actual values and forecast
        plt.figure(figsize=(12,6))
        plt.plot(y_test.index, y_test.values, label='Actual')
        plt.plot(y_test.index, y_pred_ses, label='Forecast')
        plt.legend()
        plt.title('Exponential Smoothing Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        st.pyplot(plt)

# Run the Streamlit app
if __name__ == '__main__':
    app()
