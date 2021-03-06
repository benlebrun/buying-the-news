from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import re
import sys 
import os
import csv
import uuid
import pandas as pd
from tqdm import tqdm

def get_article(url):
	''' returns soup object '''
	try:
		return BeautifulSoup(urlopen(Request(url, headers={'User-Agent': "Chrome/5.0"})), 'lxml')
	except:
		return -1

def get_text(soup):
	''' returns list of paragraphs (list of strings) '''
	article_body = soup.find('div', class_='article-body')
	paragraphs = article_body.find_all('p')
	return [p.get_text() for p in paragraphs]

def get_author(soup):
	''' returns author '''
	return soup.find('a', class_='author-name').get_text()

def get_date(soup):
	''' returns article date (string) '''
	return soup.find('time')['datetime']

def get_headline(soup):
	''' returns article headline (string) '''
	return re.sub('\n|\t', '', soup.find('div', class_='headlines').get_text())

def get_category(soup):
	''' returns article category (string) '''
	return re.sub('\n|\t', '', soup.find('div', class_='breadcrumbs').get_text())

def scrape_article(url, dest):
	''' 
	scrapes article. 
	essentially just calls all relevant functions and writes article data.

	input
	---------
	url : str
		article url
    dest : str 
		save destination

	output
	---------
	list of strings containing article metadata 
	'''
	soup = get_article(url)

	try:
		text = get_text(soup)
		date = get_date(soup)
		author = get_author(soup)
		headline = get_headline(soup)
		cat = get_category(soup)
	except:
		# try again with different url if error
		retry = url.split('/')
		url = '/'.join(retry[0:3])+'/'+retry[-1]
		soup = get_article(url)

		text = get_text(soup)
		date = get_date(soup)
		author = get_author(soup)
		headline = get_headline(soup)
		cat = get_category(soup)

	unique_id = uuid.uuid4().hex[0:5]
	str_date = date.partition(' ')[0]
	filename = str_date+'-'+unique_id+'.txt'

	# write paragraphs to text file w/ unique ID 
	with open(dest+filename, 'w+') as f: 
		for p in text: 
			if p != '': 
				f.write(p + '\n')
	
	return [url, headline, str_date, author, cat, filename]

def main():
	if len(sys.argv) != 4:
		print('Input source of url csv file, metadata destination, and file destination.')
		sys.exit()

	urls = pd.read_csv(sys.argv[1])
	meta = sys.argv[2]
	dest = sys.argv[3]

	with open(meta, mode='a+') as file:
		csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for url in tqdm(urls.url):
			try:
				# scrape, write article, and return metadata 
				article_info = scrape_article(url, dest)
				csv_writer.writerow(article_info)
			except Exception as e:
				print('Error %s for %s' % (e, url))
				continue

if __name__=='__main__':
	main()
