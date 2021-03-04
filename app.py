from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange
import pandas as pd
from flask import request, redirect
from datetime import datetime
import os
import numpy as np 
import matplotlib.pyplot as plt
from pytz import timezone
from dateutil import tz
import joblib
from datetime import timedelta



app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'mysecretkey'

#To extract respective dataframe 
df_50=pd.read_csv('forecast_nifty50.csv')
df_100=pd.read_csv('forecast_nifty100.csv')
today = datetime.today()
from_zone = tz.gettz('UTC')
to_zone = tz.gettz('Asia/Kolkata')
today=today.replace(tzinfo=from_zone)
now=today.astimezone(to_zone)
now=today.strftime('%Y%m%d')

tomo=datetime.today() + timedelta(1)
date_str=tomo.strftime('%d-%m-%Y') 
dir=os.getcwd()
dir_50='/static/images/Hourly_Forcast_Nifty50_'+now+'.png'
dir_100='/static/images/Nifty100_hourly_forecast_'+now+'.png'


@app.route('/')
def index():
	
	return render_template('sample.html')

@app.route('/prediction',methods=['POST'])
def prediction():
	nifty = request.form['nifty']
	if(nifty=='Nifty50'):	
		return redirect(url_for("prediction_50"))
	else:
		return redirect(url_for("prediction_100"))


@app.route('/prediction_50')
def prediction_50():
	var1=round(float(df_50['Forecast'].iloc[0]),2)
	var2=round(float(df_50['Forecast'].iloc[1]),2)
	var3=round(float(df_50['Forecast'].iloc[2]),2)
	var4=round(float(df_50['Forecast'].iloc[3]),2)
	var5=round(float(df_50['Forecast'].iloc[4]),2)
	var6=round(float(df_50['Forecast'].iloc[5]),2)
	var7=round(float(df_50['Forecast'].iloc[6]),2)
	print("Nifty50")
	print(dir_50)
	return render_template('predictions_n50.html',date=date_str,var1=var1,var2=var2,var3=var3,var4=var4,var5=var5,var6=var6,var7=var7,dir=dir_50)

@app.route('/prediction_100')
def prediction_100():
	var1=round(float(df_100['Forecast'].iloc[0]),2)
	var2=round(float(df_100['Forecast'].iloc[1]),2)
	var3=round(float(df_100['Forecast'].iloc[2]),2)
	var4=round(float(df_100['Forecast'].iloc[3]),2)
	var5=round(float(df_100['Forecast'].iloc[4]),2)
	var6=round(float(df_100['Forecast'].iloc[5]),2)
	var7=round(float(df_100['Forecast'].iloc[6]),2)
	print("Nifty100")
	print(dir_100)
	return render_template('predictions_n100.html',date=date_str,var1=var1,var2=var2,var3=var3,var4=var4,var5=var5,var6=var6,var7=var7,dir=dir_100)


if __name__ == '__main__':
    app.run(debug=True)
