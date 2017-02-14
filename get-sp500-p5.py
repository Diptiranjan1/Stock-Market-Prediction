import bs4 as bs
import pickle		#serializes any python object// here we serialize before saving s&p500 data
import requests


def save_sp500():
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
		tickers.append(ticker)

	with open("sp500tickers.pickle","wb") as f:
		pickle.dump(tickers,f)

	print(tickers)

	return tickers

save_sp500()
