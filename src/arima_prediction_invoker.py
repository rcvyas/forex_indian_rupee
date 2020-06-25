from arima_prediction import ArimaPrediction
import ConfigParser as cp
import argparse as ap
import sys
import os

def conf_initiator(conf_file):
	__section__ = 'info'
	config = cp.ConfigParser()
	config.read(conf_file)		
	config._sections
	config_dict = dict(config._sections[__section__])
	return  config_dict


def parse_initiator():
	parser = ap.ArgumentParser()
	parser.add_argument('-c', '--conf',action='store', dest='conf_file', help='specify --act new or pull or checkin ', required=True)
	results = parser.parse_args()
	
	try:
		conf_file=results.conf_file
	except:
		print 'no config file specified'
		parser.print_help() 
		sys.exit()
	return conf_initiator(results.conf_file)

def __main__():
	config_dict = parse_initiator()
	obj = ArimaPrediction(config_dict)
	obj.arima_model()	
	
__main__()
