from geocode import Geoparse
from tqdm import tqdm
import googlemaps
import re 
import argparse
import logging
import pickle
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def open_doc(src):
    with open(src, 'rb') as fp:
        doc = pickle.load(fp)
    return doc

def save(qs, src):
    with open(src, 'wb') as fp:
        pickle.dump(qs, fp, pickle.HIGHEST_PROTOCOL)

def load(src):
    with open(src, 'rb') as fp:
        return pickle.load(fp) 

def parse(files, gmap, dest, queries_name="queries.p"):
    for file, url in tqdm(files):
        try:
            doc = open_doc(file)
        except Exception as e:
            print(f'{file} : {e}')
            with open('errors.txt', 'a+') as f:
                f.write(f"{file} ")
            continue
        geo_file = re.sub('.p', '.txt', file.split('/')[-1])
        #if geo_file in os.listdir(dest): continue
        queries = load(queries_name)
        try:
            geo_doc = Geoparse(doc=doc, 
                               url=url, 
                               filename=os.path.join(dest, geo_file), 
                               stored_queries=queries, 
                               gmap=gmap, 
                               query=True, 
                               save=True)
        except Exception as e:
            print(f'{file} : {e}')
            continue
        queries = geo_doc.queries
        save(queries, queries_name)

def main():
    parser = argparse.ArgumentParser('Geotag parsed articles.')
    parser.add_argument('--articles', help='txt file where each line is /PATH/TO/ARTICLE URL', required=True)
    parser.add_argument('--dest', help='where to save place files', required=True)
    parser.add_argument('--queries-name', help='dict of previously saved queries', required=False, default='queries.p')
    args = parser.parse_args()

    gmap = googlemaps.Client(key='key-goes-here')

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    already = os.listdir(args.dest)
	
    with open(args.articles, 'r') as f:
        files = [[re.sub('\n', '', l) for l in line.split()] for line in f.readlines() if line != '' and re.sub('.p', '.txt', line.split()[0].split('/')[-1]) not in already]

    parse(files=files, gmap=gmap, dest=args.dest, queries_name=args.queries_name)

if __name__=='__main__':
    main()
