import pandas as pd
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib
import matplotlib.dates as mdates
from my_arima_model import ARIMA
import random as rd

class ArimaPrediction:
	
	def __init__(self, config_dict):

		self.CURRENCY_X = config_dict['currency_x']
		self.CURRENCY_Y = config_dict['currency_y']
		self.FORECAST = int(config_dict['forecast'])
		self.INTERVAL = int(config_dict['interval'])
		train_set='data/train/'+self.CURRENCY_X+'-'+self.CURRENCY_Y+'_1997-2016.csv'
		df_train = pd.read_csv(train_set)
		test_set='data/test/'+self.CURRENCY_X+'-'+self.CURRENCY_Y+'_JAN-OCT.csv'
		df_test = pd.read_csv(test_set)
		self.train_dates, self.train_prices = self.get_dates_and_prices(df_train)
		self.test_dates, self.test_prices = self.get_dates_and_prices(df_test)
		
	def get_dates_and_prices(self,df):
		dates = list(df['Date'])
		prices = [float(str(x).replace(',', '')) for x in list(df['Price'])]
		#prices = list(df['Price'])
		return dates, prices
		
	def my_plotter(self, X, y, INTERVAL, LABEL, COLOR):
		
		plt.ylabel(self.CURRENCY_Y+'/'+self.CURRENCY_X)
		
		plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
		plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=INTERVAL))
		
		
		X = mdates.datestr2num(X)
		plt.plot(X, y, color=COLOR, label=LABEL)
		
		plt.xlabel('Time Period')
		plt.title(self.CURRENCY_X+' vs '+self.CURRENCY_Y)
		plt.gcf().autofmt_xdate()
		plt.legend(loc='upper left')
		
		
	def arima_model(self):
		y = self.train_prices
		init_val = y[0]
		
		'''
		X = self.date_parser(self.train_dates)	
		self.my_plotter(X, y, 730, 'Original dataset', 'blue')
		plt.show()
		sys.exit()
		'''
		
		model = ARIMA(y, order=(4, 1, 1))
		results_AR = model.fit(disp=-1)

		y = [float(x) for x in list(results_AR.fittedvalues)]
		y[-1] = init_val
		for i in range(len(y)-2, -1,-1):
			y[i] += y[i+1]
		#	print 'lengths x, y:', len(X), len(y), '\n\n'
		y = [y[-1]]+y[::-1]
		y = y[::-1]
		history = y[:]
		history = history+list(self.test_prices)[::-1]
		#print history
		#sys.exit()
		observations = self.test_prices
		predictions = list()
		#	print 'init:',y[:10], ', final:', y[-10:] 
		X, future = self.date_parser(self.test_dates, self.FORECAST)
		#print self.test_dates
		#sys.exit()
		for t in range(len(future)):#len(observations)):
			model = ARIMA(history, order=(2,1,1))
			model_fit = model.fit(disp=0)
			output = model_fit.forecast()
			pred = output[0]
			predictions.append(float(pred))
			obs = observations[t]
			#history.append(obs)
			history.append(history[-(t+rd.randint(1, self.INTERVAL))])
			#	print 'predicted=%f, expected=%f' % (pred, obs)
			#predictions = output[:len(observations)]
			
			#	obs = test[t]
			#	history.append(obs)
			#	print('predicted=%f, expected=%f' % (np.exp(pred), np.exp(obs))
		#X, future = self.date_parser(self.test_dates, self.FORECAST)	

		#self.my_plotter(future, observations, self.INTERVAL, 'Original dataset', 'blue')
		self.my_plotter(future, predictions, self.INTERVAL, 'Prediction dataset', 'red')
		#	plt.plot(X, y, color='red', label='Prediction dataset')
		
		plt.show()
		
		
		
	def date_parser(self, date_list, forecast):
		months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
		new_date_list = list()	
		for x in date_list:
			month, day, year = x.split(' ')
			day = day.strip(',')
			month = months.index(month)+1
			day, year = int(day), int(year)
			formatted_date = str(dt.datetime(year, month, day)).split(' ')[0]
			new_date_list.append(formatted_date)
		year, month, day = new_date_list[0].split('-')
		year, month, day = int(year), int(month), int(day)
		date = dt.datetime(year,month,day,12,4,5)
		future_date_list = []
		for i in range(forecast): 
			date += dt.timedelta(days=1)
			future_date_list.append(str(date).split(' ')[0])
		return new_date_list, future_date_list	
