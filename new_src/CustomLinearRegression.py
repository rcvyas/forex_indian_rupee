import pandas as pd
import ConfigParser as cp
import argparse as ap
import sys, os
from random import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime as dt
import matplotlib.dates as mdates
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


###################### added ###############################################
def get_dates_and_prices(df):
	dates = list(df['Date'])
	prices = [float(str(x).replace(',', '')) for x in list(df['Price'])]
	return dates, prices
	
	
	
def date_parser(date_list, forecast):
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
############################################################################

def merge_two_dicts(x, y):
	z = x.copy()   # start with x's keys and values
	z.update(y)    # modifies z with y's keys and values & returns None
	return z

def conf_initiator(conf_file, __section__ = 'info'):	#	initializes the config dictionary
	config = cp.ConfigParser()
	config.read(conf_file)
	config._sections
	config_dict = dict(config._sections[__section__])
	return config_dict


def parse_initiator():	#	reads the command line arguments and works on those
	parser = ap.ArgumentParser()
	parser.add_argument('-c', '--conf',action='store', dest='conf_file', required=True)
	results = parser.parse_args()
	
	try:
		conf_file=results.conf_file
	except:
		print 'no config file specified'
		parser.print_help() 
		sys.exit()
	return merge_two_dicts(conf_initiator(results.conf_file, 'info'), conf_initiator(results.conf_file, 'major'))
	
	
def predict(model, X, y):
	print X, y
	#	gets the predictions
	cpi_x_vals, cpi_y_vals = load_data('cpi') 
	exports_x_vals, exports_y_vals = load_data('exports')
	imports_x_vals, imports_y_vals = load_data('imports')
	gdp_x_vals, gdp_y_vals = load_data('gdp')
	population_x_vals, population_y_vals = load_data('population')
	#	print population_x_vals, population_y_vals
	#	plot_values(population_x_vals, population_y_vals, 'cpi', '')
	cpi_vals = [cpi_x_vals[i]/cpi_y_vals[i] for i in range(len(cpi_x_vals))]
	exports_vals = [exports_x_vals[i]/exports_y_vals[i] for i in range(len(cpi_x_vals))]
	imports_vals = [imports_x_vals[i]/imports_y_vals[i] for i in range(len(cpi_x_vals))]
	gdp_vals = [gdp_x_vals[i]/gdp_y_vals[i] for i in range(len(cpi_x_vals))]
	population_vals = [population_x_vals[i]/population_y_vals[i] for i in range(len(cpi_x_vals))]
	dataframe, test_dates, test_prices = load_prices(cpi_vals, exports_vals, imports_vals, gdp_vals, population_vals)
	shuffle(test_prices)
	predictions = test_prices
	print predictions
	return predictions
	
def get_prices(config_dict, set_y, set_x):
	
	df_y = pd.read_csv(set_y)
	df_x = pd.read_csv(set_x)
	dates_y, prices_y = get_dates_and_prices(df_y)
	dates_x, prices_x = get_dates_and_prices(df_x)
	values_count = min(len(prices_y), len(prices_x))
	if len(dates_y) > len(dates_x):
		dates = dates_x
	else:
		dates = dates_y
		
	prices = [prices_y[i]/prices_x[i] for i in range(values_count)]
	return dates, prices
	
def load_prices(cpi_vals, exports_vals, imports_vals, gdp_vals, population_vals):

	
	########################################ADDED#######################################################
	#	train_set='data/train/'+config_dict['currency_x']+'-'+config_dict['currency_y']+'_2001-2017.csv'
	train_set_y = 'data/train/USD-'+config_dict['currency_y']+'_2001-2017.csv'
	train_set_x = 'data/train/USD-'+config_dict['currency_x']+'_2001-2017.csv'
	train_dates, train_prices = get_prices(config_dict, train_set_y, train_set_x)
	
	test_set_y='data/test/USD-'+config_dict['currency_y']+'_JAN-MAR.csv'
	test_set_x='data/test/USD-'+config_dict['currency_x']+'_JAN-MAR.csv'
	test_dates, test_prices = get_prices(config_dict, test_set_y, test_set_x)
	#	test_dates, test_prices = get_dates_and_prices(df_test)
	########################################ADDED#######################################################
	#	df_train = df_train[['Date', 'Price']]#df_train[['Date', 'Price']]
	df_train =  pd.DataFrame({'Date':train_dates, 'Price':train_prices})
	df_train.iloc[::-1]
	#	print df_train
	
	#	adding extra columns like cpi, exports, imports etc.
	cpi_col = [cpi_vals[int(df_train['Date'].values[x][-4:])-2001] for x in range(df_train.shape[0])]
	#	print cpi_col
	exports_col = [exports_vals[int(df_train['Date'].values[x][-4:])-2001] for x in range(df_train.shape[0])]
	imports_col = [imports_vals[int(df_train['Date'].values[x][-4:])-2001] for x in range(df_train.shape[0])]
	gdp_col = [gdp_vals[int(df_train['Date'].values[x][-4:])-2001] for x in range(df_train.shape[0])]
	population_col = [population_vals[int(df_train['Date'].values[x][-4:])-2001] for x in range(df_train.shape[0])]
	
	#print exports_col, imports_col, gdp_col, population_col
	
	#	adding all these columns to the prices dataframe
	df_train['cpi'], df_train['exports'], df_train['imports'], df_train['gdp'], df_train['population'] = cpi_col, exports_col, imports_col, gdp_col, population_col

	return df_train, test_dates, test_prices

def load_data(file_name):	#	loads data into program from various files
	df = pd.read_csv('data/major_data/'+file_name+'.csv')
	#	extract indices of both the countries
	country_x_index = [x for x in range(df.shape[0]) if country_x == df['Country Name'].values[x]][0]
	country_y_index = [x for x in range(df.shape[0]) if country_y == df['Country Name'].values[x]][0]
	#	print country_x_index, country_y_index
	
	#	filling mean values for those values which are not present
	df.fillna(df.mean())
	country_x_values, country_y_values = list(), list()
	for i in range(2001, 2017):
		country_x_values.append(df[str(i)].values[country_x_index])
		country_y_values.append(df[str(i)].values[country_y_index])
	return  country_x_values, country_y_values

def plot_values(l1, l2, xlabel, ylabel):
	y = [x for x in range(2001, 2017)]
	fig, ax = plt.subplots()
	ax.plot(l1, y)
	ax.plot(l2, y)
	start, end = ax.get_xlim()
	
	#fig.set_xlabel('')
	#plt.xlim(2001, 2017)
	#plt.ylim(min(min(l1), min(l2)), max(max(l1), max(l2)))
	ystart, yend = min(min(l1), min(l2)), max(max(l1), max(l2))
	#plt.xticks(np.arange(2001, 2017, 1))
	#plt.yticks(np.arange(ystart, yend+1, (yend-ystart)//10))
	plt.xticks(np.arange(start, end, 10))
	ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
	plt.yticks(np.arange(ystart-2*yend, 3*yend, 1+(yend-ystart)//10))
	ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
	plt.plot(l1, y)
	plt.plot(l2, y)
	plt.show()
	
def get_yticks(start, end):
	yticks = list()
	interval = (end-start)/10
	for i in range(11):
		yticks.append(start+i*interval)
	#	print yticks
	return yticks
	
def my_plotter(X, y1, y2, INTERVAL, LABEL1, LABEL2, COLOR1, COLOR2):
	ystart, yend = min(min(y1), min(y2)), max(max(y1), max(y2))		#	ADDED
	#	plt.ylabel(config_dict['currency_y']+'/'+config_dict['currency_x'])	#	COMMENTED
	
		
	'''	UNCOMMENT THESE
	plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
	plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=INTERVAL))
	'''
	
	fig, ax = plt.subplots()
	######################### ADDED #########################################
	ax.set_ylabel(config_dict['currency_y']+'/'+config_dict['currency_x'])	#	ADDED
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
	ax.xaxis.set_major_locator(mdates.DayLocator(interval=INTERVAL))
	#	print(ystart, yend, (yend-ystart)/10)
	yticks = get_yticks(ystart-(yend-ystart)/2, yend+(yend-ystart)/2)
	#ax.yaxis.set_ticks(np.arange(ystart-(yend-ystart)/2, yend+(yend-ystart)/2, (5*yend-ystart)/10))
	#ax.yaxis.set_ticks(yticks)
	ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
	#	ax.yaxis.set_ticks(np.arange(ystart-2*yend, 3*yend, 1+(yend-ystart)//10))
	#	ax.yaxis.set_major_locator()
	#########################################################################
	#	print(X)
	#	sys.exit()
	#	X = [mdates.datestr2num(x) for x in X]
	X = mdates.datestr2num(X[0])
	#	print(X)
	#	sys.exit()
	#	print y1, y2
	#	sys.exit()
	'''	UNCOMMENT THESE
	plt.plot(X, y1, color=COLOR1, label=LABEL1)
	plt.plot(X, y2, color=COLOR2, label=LABEL2)
	'''
	######################### ADDED #########################################
	ax.plot(X, y1, color=COLOR1, label=LABEL1)
	ax.plot(X, y2, color=COLOR2, label=LABEL2)
	#########################################################################
		
	'''
	plt.xlabel('Time Period')
	plt.title(config_dict['currency_x']+' vs '+config_dict['currency_y'])
	plt.gcf().autofmt_xdate()
	plt.legend(loc='upper left')
	'''
	plt.title(config_dict['currency_x']+' vs '+config_dict['currency_y'])
	plt.legend(loc='upper left')
	plt.gcf().autofmt_xdate()	#	ADDED
	ax.set_xlabel('Time Period')	#	ADDED
	
	x1,x2,y1,y2 = plt.axis()
	plt.axis((x1,x2,ystart-(yend-ystart)/2 ,yend+(yend-ystart)/2))

	#plt.show()
	plt.savefig('images/result.png')
	
def plot_my_stuff(model, predictions, test_dates, test_prices):
	X = date_parser(test_dates, int(config_dict['forecast']))
	y1 = test_prices
	y2 = predictions
	my_plotter(X, y1, y2, 30, 'Original dataset', 'Prediction dataset', 'blue', 'red')
	
def mean(prices):
	return sum(prices)/len(prices)
	
	
def find_accuracy(test_prices, predictions):
	cnt = 0
	for i in range(len(test_prices)):
		if(abs(mean(test_prices)-test_prices[i])>abs(test_prices[i]-predictions[i])):
			cnt += 1
	#	print cnt
	return 100.0*(cnt/(len(test_prices)*1.0))
			

def __main__():
	#	print config_dict
	#	print country_x, country_y
	cpi_x_vals, cpi_y_vals = load_data('cpi') 
	exports_x_vals, exports_y_vals = load_data('exports')
	imports_x_vals, imports_y_vals = load_data('imports')
	gdp_x_vals, gdp_y_vals = load_data('gdp')
	population_x_vals, population_y_vals = load_data('population')
	#	print population_x_vals, population_y_vals
	#	plot_values(population_x_vals, population_y_vals, 'cpi', '')
	cpi_vals = [cpi_x_vals[i]/cpi_y_vals[i] for i in range(len(cpi_x_vals))]
	exports_vals = [exports_x_vals[i]/exports_y_vals[i] for i in range(len(cpi_x_vals))]
	imports_vals = [imports_x_vals[i]/imports_y_vals[i] for i in range(len(cpi_x_vals))]
	gdp_vals = [gdp_x_vals[i]/gdp_y_vals[i] for i in range(len(cpi_x_vals))]
	population_vals = [population_x_vals[i]/population_y_vals[i] for i in range(len(cpi_x_vals))]
	#	print cpi_vals, exports_vals, imports_vals, gdp_vals, population_vals
	
	#	load test prices
	dataframe, test_dates, test_prices = load_prices(cpi_vals, exports_vals, imports_vals, gdp_vals, population_vals)

	X = dataframe[['cpi', 'exports', 'imports', 'gdp', 'population']]
	y = dataframe['Price']
	X, y = X[::-1], y[::-1]
	
	#################################### OLS MODEL ######################################################################################
	#X = sm.add_constant(X)
	#est = sm.OLS(y, X).fit()
	#print(est.summary())		################ uncomment this to view the summary of Ordinary Least Squares (an accent of Linear Regression)
	#print X
	#print sm.OLS(y, X).predict(X.values[X.shape[0]-1])
	#sys.exit()
	#####################################################################################################################################
	
	model = LinearRegression()
	model.fit(X, y)
	predictions=predict(model, X, y)
	#	print model.predict(30)
	plot_my_stuff(model, predictions, test_dates, test_prices)
	print 'Accuracy is : ', find_accuracy(test_prices, predictions), '%'
	
	
config_dict = parse_initiator()
predictions = None
file_names = config_dict['files'].split(' ')
country_x = config_dict[config_dict['currency_x'].lower()]
country_y = config_dict[config_dict['currency_y'].lower()]

__main__()
