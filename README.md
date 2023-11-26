# Stock_prediction_app-using_LSTM
 https://nse-predictions-app.herokuapp.com/ Intra day trading involves buying and selling of stocks within the same trading day. Here stocks are purchased, not with an intention to invest, but for the purpose of earning profits by harnessing the movement of stock indices. The following app predicts the weighted average stock prices of top 50 (Nifty50 ) and top 100 (Nifty100) companies on an hourly basis everyday. 


# Stock Market Forecasting Web App

## Overview

This Flask web application provides hourly forecasts for Nifty50 and Nifty100 indices. It utilizes machine learning models to predict future values and displays the results on a user-friendly interface.

## Project Structure

- **app.py**: The main Flask application script containing the web app logic.
- **templates/**: Folder containing HTML templates for rendering web pages.
- **static/images/**: Folder to store generated forecast images.

## Requirements

- Flask
- Flask-WTF
- pandas
- numpy
- matplotlib
- pytz
- joblib

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app:**
   ```bash
   python app.py
   ```
   The app will be accessible at [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

## Web Pages

### Home Page (`sample.html`)

The main landing page of the web app.

### Forecast Pages

- **/prediction**: Accepts form input to choose between Nifty50 and Nifty100.
- **/prediction_50**: Displays hourly forecasts for Nifty50.
- **/prediction_100**: Displays hourly forecasts for Nifty100.

## Data Sources

The forecasting models are trained on historical stock market data.

## Acknowledgements

- The Flask web application is built using Flask, Flask-WTF, pandas, numpy, and matplotlib.
- Hourly forecasts are generated using machine learning models trained on historical data.
- Images are saved to the `static/images/` folder for display in the web app.

Feel free to explore the code in `app.py` and the associated HTML templates for more details.

**Note:** Adjust file paths and comments as needed for your project structure.
