from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import re, sys, os, csv, uuid
import pandas as pd
from tqdm import tqdm

def get_article(url):
	return BeautifulSoup(urlopen(Request(url, headers={'User-Agent': "Chrome/5.0"})), 'lxml')

def get_text(soup):
	return [p.get_text() for p in soup.find('div', class_='article-body').find_all('p')]

def get_date(soup):
	return soup.find('time')['datetime']

def get_author(soup):
	return soup.find('a', class_='author-name').get_text()

def get_headline(soup):
	return re.sub('\n|\t', '', soup.find('div', class_='headlines').get_text())

def get_category(soup):
	return re.sub('\n|\t', '', soup.find('div', class_='breadcrumbs').get_text())

def scrape_article(url, dest):
	soup		  = get_article(url)
	t, d, a, h, c = get_text(soup), get_date(soup), get_author(soup), get_headline(soup), get_category(soup)
	unique_id	  = uuid.uuid4().hex[0:5]
	str_date	  = d.partition(' ')[0]
	filename	  = str_date+'-'+unique_id+'.txt'

	with open(dest+filename, 'w+') as f: 
		for p in t: 
			if p != '': 
				f.write(p + '\n')
	
	return [url, h, str_date, a, c, filename]

def main():
	if len(sys.argv) != 4:
		print('Input url csv file, metadata destination, and file destination.')
		sys.exit()

	urls = pd.read_csv(sys.argv[1])
	meta = sys.argv[2]
	dest = sys.argv[3]

	with open(meta, mode='a+') as file:
		csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		for url in tqdm(urls.url[7886:]):
			try:
				article_info = scrape_article(url, dest)
				csv_writer.writerow(article_info)
			except Exception as e:
				print('Error %s for %s' % (e, url))
				continue

if __name__=='__main__':
	main()


