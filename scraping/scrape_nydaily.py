'''
2020/09

Script used to scrape nydaily articles. Input path to file containing 
article urls as well as meta data destination and article content destination. 

'''
from urllib.request import Request, urlopen
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import sys, os, csv
from tqdm import tqdm
import uuid
from datetime import datetime

def parse_ps(main):
    ''' returns article paragraphs and links to external webpages '''
    text, hrefs = [], []
    for p in main.find_all(['p', 'h1', 'h2', 'h3']):
        text.append(p.get_text())
        links = p.find_all('a')
        if len(links) > 0:
            hrefs.extend([h['href'] for h in links])
    return text, hrefs

def get_article(url): 
    ''' returns article meta data and content '''
    hdr  = {'User-Agent': "Chrome/5.0"}
    req  = Request(url,headers=hdr)
    page = urlopen(req)
    soup = BeautifulSoup(page, 'lxml')
    try:
        main = soup.find('div', 
                         class_='wrapper clearfix pb-curated full pb-feature pb-layout-item pb-f-article-body')
        text, hrefs = parse_ps(main)
    except AttributeError:
        main = soup.find('div', 
                         class_='wrapper clearfix col pb-curated full pb-feature pb-layout-item pb-f-article-body')
        text, hrefs = parse_ps(main)
    try:
        author = soup.find('a', rel='author').get_text()
    except AttributeError:
        try:
            author = soup.find_all('span', class_='uppercase')[0].get_text()
        except IndexError:
            author = soup.find_all('span', class_='timestamp timestamp-article')[0].get_text()
    
    headline = soup.find('h1', 
                         class_='spaced spaced-xl spaced-top spaced-bottom abdp-headline hdln-nydn').get_text()
    date = soup.find_all('span', class_='timestamp timestamp-article')[1].get_text()
    if date == 'at': 
        date = soup.find_all('span', class_='timestamp timestamp-article')[0].get_text()
    author = author.lower()
    
    return text, headline, author, date, hrefs

def scrape_article(url, dest):
    ''' scrapes article and writes to txt file. also returns article meta data '''
    text, headline, author, date, hrefs = get_article(url)
    unique_id = uuid.uuid4().hex[0:5]
    date = datetime.strptime(date, ' %b %d, %Y ')
    str_date = date.strftime('%Y-%m-%d')
    filename = str_date+'-'+unique_id+'.txt'

    if date >= datetime(2012,1,1):
        with open(dest+filename, 'w+') as f:
            for p in text:
                f.write(p + '\n')
    else:
        filename = 'N/A'

    if len(hrefs) > 0: 
        hrefs = ', '.join(hrefs)
    else: 
        hrefs = ''
    
    return [url, headline, str_date, author, hrefs, filename]

def main():
    if len(sys.argv) != 4:
        print('Input url csv file, metadata destination, and file destination.')
        sys.exit()

    urls = pd.read_csv(sys.argv[1])
    meta = sys.argv[2]
    dest = sys.argv[3]

    try:
        already_collected = pd.read_csv(meta)
        to_scrape  = list(set(urls.url) - set(already_collected.url))
        print(f'Already collected {len(urls.url) - len(to_scrape)} articles.')
    except Exception as e:
        print(e)
        to_scrape = urls.url

    with open(meta, mode='a+') as file:
        csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([])
        for url in tqdm(to_scrape):
            try:
                article_info = scrape_article(url, dest)
                csv_writer.writerow(article_info)
            except Exception as e:
                print('Error %s for %s' % (e, url))
                continue
    file.close()

if __name__=='__main__':
    main()











