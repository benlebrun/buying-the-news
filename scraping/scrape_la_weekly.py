from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import pandas as pd
import re
import sys, os
import requests
import wget, csv
import time
from selenium import webdriver

def get_article_urls(soup, year): 
    '''
    gets LAWEEKLY article urls from page of a given year
    e.g. https://www.laweekly.com/2019/page/45/
    '''
    articles = soup.find_all('article')
    hrefs = []
    for article in articles:
        try:
            article_yr = int(article.find('span', class_='cb-date').find('time')['datetime'].partition('-')[0])
        except AttributeError:
            continue
        if article_yr != int(year):
            continue
        a_s = article.find_all('a')
        hrefs.append(a_s[0]['href'])
    
    return hrefs

def get_article(driver, url, dest, mfilepath, filename, exe_path = '../chromedriver'):
    '''
    This function scrapes LA WEEKLY articles.
    Specify url, file dest, metadata dest, 
    and path to driver executable. 
    We must use a driver here since articles are hosted
    on a dynamic webpage. 
    '''
    driver.get(url)
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source)
    try:
        ps = soup.find('section', class_='cb-entry-content clearfix').find_all('p')
    except:
        ps = soup.find('div', class_='post-wrapper').find_all('p')
    aid = soup.find('article')['id']

    try:
        # some article headlines are formatted differently
        title = soup.find('h1',class_='entry-title cb-entry-title entry-title cb-title entry_test2').get_text()
    except AttributeError:
        title = soup.find('h1',class_='entry-title cb-entry-title entry-title cb-title').get_text()
    print(title) 
    name = soup.find_all('meta', itemprop='name')[1]['content']
    pub_date = soup.find_all('meta', itemprop='datePublished')[0]['content']
    author = soup.find('span',class_='cb-author vcard author').get_text()

    try:
        assert name == author
    except AssertionError:
        author = np.nan
        
    fname = dest + filename #str(aid).partition('-')[2] + '.txt'
    f = open(fname,'w+')
    for p in ps:
        f.write(str(p.get_text()) + '\n')
    f.close()

    fname = filename #str(aid).partition('-')[2] + '.txt'
    with open(mfilepath, mode='a+') as file:
        csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([url, title, author, pub_date, fname])
        file.close()
    #driver.close()

if __name__ == '__main__':

    file = sys.argv[1]
    meta = sys.argv[2]
    urls = pd.read_csv(file)
    exe_path = '../chromedriver'
    driver = webdriver.Chrome(executable_path=exe_path)
    for url,f in zip(urls.url, urls.file):
        try:
            print(url)
            get_article(driver, url, '../data/la/articles/', meta, f)
        except Exception as e:
            print("ERROR: %s while scraping %s" % (str(e),url))
    driver.close()

