import bs4 as bs
import pickle		#serializes any python object// here we serialize before saving s&p500 data
import requests
import datetime as dt 
import pandas as pd 
import pandas_datareader.data as web
import os

def save_sp500_ticker():
	resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
	#soup is the object that comes from BeautifulObject
	soup = bs.BeautifulSoup(resp.text, 'lxml')
	#table == will find table data// should be specified
	table = soup.find('table',{'class':'wikitable sortable'})
	tickers = []
	#tr = table row, td = table data
	#for each table row
	for row in table.findAll('tr')[1:]:
		ticker = row.findAll('td')[0].text # we want the 0th column, we want the text from soupobject
		# get past BRK...
		mapping = str.maketrans(".","-") 
		ticker = ticker.translate(mapping)
		
		tickers.append(ticker)

	with open("sp500tickers.pickle","wb") as f:
		pickle.dump(tickers,f)

	print(tickers)

	return tickers

#save_sp500_ticker()

def get_data_from_yahoo(reload_sp500=False):
	if reload_sp500:
		tickers = save_sp500_ticker()
	else:
		with open("sp500tickers.pickle","rb") as f:
			tickers = pickle.load(f)

	# Now we will save all SP500 data as csv files
	if not os.path.exists('stock_dfs'):	# if this directory doesn't exist
		os.makedirs('stock_dfs')

	start =dt.datetime(2000,1,1)
	end = dt.datetime(2017,2,8)

	# if net is slow just put in parameter in tickers[:?] to specify number of companies you want etail of.
	for ticker in tickers:
		print(ticker)
		if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
			df = web.DataReader(ticker,'yahoo',start,end)
			df.to_csv('stock_dfs/{}.csv'.format(ticker))
		else:
			print('Already have {}'.format(ticker))

get_data_from_yahoo()			
# Due throttling this whole data scrapping may stop after certain
# amount of time. Use 'sleep' to pause and continue after that...