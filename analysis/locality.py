import pickle
import pandas as pd
import os
import math
import numpy as np
from itertools import chain
from tqdm.notebook import tqdm
import geopy.distance
try:
    from geocode import Geoparse
except Exception as e:
    print(f'Failed to load Geoparse : {e}')
from random import sample
from collections import Counter
gmap = False

############################
### preamble

'''
with open('queries.p', 'rb') as f:
    queries = pickle.load(f)

'''
with open('pub2loc.p', 'rb') as f:
    pub2loc = pickle.load(f)  

def openDoc(name):
    with open(name, 'rb') as fp:
        return pickle.load(fp)
    
def openDocs(names):
    return [openDoc(doc) for doc in tqdm(names)]

def get_text(doc):
    return ' '.join(list(chain(*[[t['text'] for t in s] for s in doc])))

def get_state(loc):
    ''' Takes in a result and returns state '''
    try:
        return [part['short_name'] for part in loc['address_components'] 
            		if part['types'] == ['administrative_area_level_1', 'political']][0]
    except IndexError:
        return -1
    
def get_country(loc):
    ''' Takes in a result and returns country '''
    try:
        return [part['short_name'] for part in loc['address_components'] 
            		if part['types'] == ['country', 'political']][0]
    except IndexError:
        return -1

def resolve(results, pub, pub2loc):
    ''' Takes in an entity mapped to multiple results
        and selects the one whose state matches the
        publication's. If both are from the same state,
        selects first result. '''
    pub_loc = pub2loc[pub]
    pub_state = get_state(pub_loc)
    result_states = [get_state(res) for res in results['result']]
    if result_states.count(pub_state) == 0:
        return {'entity': results['entity'],
                'result': [results['result'][0]]}
    else:
        return {'entity': results['entity'],
                'result': [results['result'][result_states.index(pub_state)]]}

def filter(result, pub, pub2loc=pub2loc):
    ''' takes in result and filters it '''
    import re
    # remove common NER false positives
    days = r"Mon\.|Tue\.|Wed\.|Thu\.|Fri\.|Sat\.|Sun\."
    days2 = r"^\s?Mon$|^\s?Tue$|^\s?Wed$|^\s?Thu$|^\s?Fri$|^\s?Sat$|^\s?Sun$"
    pols = r'^\s?(R\.?\-)|^\s?(D\.?\-)' # stuff like R-Arizona sometimes gets matched. we want to avoid this.
    if bool(re.search(days, result['entity'])) or bool(re.search(pols, result['entity'])) or bool(re.search(days2, result['entity'])):
        return []
    elif len(result['result']) != 1: 
        return resolve(results=result, pub=pub.split('/')[2], pub2loc=pub2loc)
    else:
        return result 

def articles2places(articles, file2url, id2place, queries, entity_types='GPE|LOC'):
    ''' Input is a list of articles (path to doc file) and returns list of lists containing
        gmap results for each entity in each article. '''
    all_places = []
    for f in tqdm(articles):
        if f[-1] != 'p':
            doc = openDoc(f[:-3]+'p')
        else:
            try:
                doc = openDoc(f)
            except Exception as e:
                print(e, ':', f)
        pub = file2url[f.split('/')[-1].split('.')[0]]
        try:
            geodoc = Geoparse(doc, pub, 'were not writing', 
                          queries, gmap, query=False, save=False)
            places = []
            for entity, ID in geodoc.get_places(entity_types, return_ents=True):
                if ID == -1:
                    continue
                place = id2place[ID].copy()
                place['entity'] = entity
                places.append(filter(place, pub=pub))
            places = [p for p in places if p != []]
        except Exception as e:
            print(e, ':', f)
            continue
        all_places.append([pub, places])
    return all_places

def places2filtered(places, state):
    return [[url, article2filtered(article, state)] for url, article in places]

def article2filtered(article, state):
    return [place for place in article if get_state(place['result'][0]) == state]

def places2ids(places):
    for place in places:
        for p in place[1]:
            if len(p['result']) == 1:
                place_ids.append(p['result'][0]['place_id'])
            else:
                place_ids.append(resolve(p, place[0].split('/')[2])['result'][0]['place_id'])
    return place_ids

def places2coords(places, approx=False):
    if approx:
         return [place2coords(place) 
                for place in places]
    else:
        return [place2coords(place) 
                for place in places if place['result'][0]['geometry']['location_type'] != 'APPROXIMATE']
    
def place2coords(place):
    loc = place['result'][0]['geometry']['location']
    return [loc['lng'], loc['lat']]

def week2places(week):
    return list(chain(*[places for url, places in week]))

def cohend(d1, d2):
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    s = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)
    return (u1 - u2) / s

############################
### frequency

def places2freqs(places, s_size=None, spectrum=True):
    ''' returns places frequency distribution given a list of places '''
    from collections import Counter
    from random import sample
    all_places = []
    for place in places:
        for p in place[1]:
            if len(p['result']) == 1:
                all_places.append(p['result'][0]['place_id'])
            else:
                all_places.append(resolve(p, place[0].split('/')[2])['result'][0]['place_id'])
    if s_size is None:
        s_size = len(all_places)
    all_places = sample(all_places, k=s_size)
    if spectrum:
        return list(chain(*[[k for _ in range(v)] for k,v in Counter(Counter(all_places).values()).items()]))
    else:
        return Counter(all_places)

def articles2freqs(articles):
    ''' list of articles -> frequencies of place_ids'''
    return list(Counter(list(chain(*[[place['result'][0]['place_id'] for place in article] 
                                                    for url, article in articles]))).values())
def periods2entropy(periods):
    '''dict of periods -> place_id entropy for each period '''
    return np.array([[date, scipy.stats.entropy(articles2freqs(articles))] for date, articles in periods.items()])

def gini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    g = 0.5 * rmad
    return g

def to_pmf(outcomes):
    cnts=Counter(outcomes)
    return np.asarray(list(cnts.values()))/sum(cnts.values())

def places2zipf(places, s_size, return_counter=False):
    from collections import Counter
    all_places = []
    for place in places:
        for p in place[1]:
            if len(p['result']) == 1:
                all_places.append(p['result'][0]['place_id'])
            else:
                all_places.append(resolve(p, place[0].split('/')[2])['result'][0]['place_id'])
    if s_size is None:
        s_size = len(all_places)
    all_places = sample(all_places, k=s_size)
    if return_counter:
        return Counter(all_places)
    else:
        return list(chain(*[[rank+1 for _ in range(v[1])] for rank,v in enumerate(Counter(all_places).most_common())]))

def articles2freqs(articles):
    ''' list of articles -> frequencies of place_ids'''
    return list(Counter(list(chain(*[[place['result'][0]['place_id'] for place in article] 
                                                    for url, article in articles]))).values())
def periods2entropy(periods):
    '''dict of periods -> place_id entropy for each period '''
    import scipy
    return np.array([[date, scipy.stats.entropy(articles2freqs(articles))] for date, articles in periods.items()])

def places2levels(places, place2level):
    levels = {'city':[], 'state':[], 'county':[], 'national':[], 'international':[]}
    for place in places:
        levels[place2level(place)].append(place)
    return levels

def places2ids(places):
    return [place['result'][0]['place_id'] for place in places]

def get_level_entropy(periods, level, place2level):
    from scipy.stats import entropy
    return [entropy(to_pmf(places2ids(places2levels(week2places(week), place2level)[level]))) for date, week in periods.items()]

def get_level_mass(periods, level, place2level):
    weights = []
    for date, week in periods.items():
        levels = {level:len(places) for level, places in places2levels(week2places(week), place2level).items()}
        if sum(levels.values()) != 0:
            weights.append(levels[level]/sum(levels.values()))
    return weights



############################
### distance

def places2distance(places, pub, pub2loc):
    ''' Takes in list of places and a publication and returns distances 
    from places to publication. '''
    import geopy.distance
    distances = []
    for place in places:
        place_loc = place['result'][0]['geometry']['location'].values()
        pub_loc = pub2loc[pub.split('/')[2]]['geometry']['location'].values()
        distances.append(geopy.distance.geodesic(tuple(pub_loc), tuple(place_loc)).km) 
    return distances

def articles2distance(articles):
    ''' Takes in list of list of 1) the pub url and 2) a list of places.
    returns the distances for each article in articles. Essentially just calling places2distance
    for a list of articles. '''
    return [places2distance(places, url, pub2loc) for url, places in tqdm(articles)]

def article2proximity(places, pub, pub2loc):
    ''' takes in list of place objects and calculates pairwise distance '''
    if len(places) <= 1: return []
    distances, i = [], 0
    for place1 in places:
        place1_loc = resolve(place1, pub=pub.split('/')[2],  pub2loc=pub2loc)['result'][0]['geometry']['location'].values()
        for place2 in places[i:]:
            place2_loc = resolve(place2, pub=pub.split('/')[2],  pub2loc=pub2loc)['result'][0]['geometry']['location'].values()
            if place1 == place2:
                continue
            distances.append(geopy.distance.geodesic(tuple(place1_loc), tuple(place2_loc)).km)
        i+=1
    distances = np.array(distances)
    return distances[~np.isnan(distances)]

def articles2proximity(articles):
    return [np.nanmedian(article2proximity(places, url, pub2loc)) for url, places in articles]

def periods2proximity(periods):
    '''dict of periods -> place_id entropy for each period '''
    return np.array([[date, np.nanmean(articles2proximity(articles))] for date, articles in tqdm(periods.items())])

def periods2distance(periods):
    '''dict of periods -> place_id entropy for each period '''
    return np.array([[date, articles2distance(articles)] for date, articles in tqdm(periods.items())])


############################
### boolean locality

def article2distances(places, city_loc):
    distances = []
    for place in places:
        if len(place['result']) > 1:
            place = resolve(place, pub=pub)
        place_loc = place['result'][0]['geometry']['location'].values()
        distances.append(get_dist(place_loc, city_loc))
    return distances

def articles2distances(articles, city_loc):
    return [[url, article2distances(article, city_loc(url))] for url, article in articles]

def periods2distances(periods, city_loc):
    ''' 
    city_loc is a fct from url -> loc (simply ignore input in non mng cases)
    i.e. sometimes the city location depends on url
    '''
    return {date:articles2distances(articles, city_loc)
                     for date, articles in periods.items() if len(articles) > 0}

def get_city(result):
    ''' result dict -> city '''
    try:
        return [part['short_name'] for part in result['address_components'] 
                    if part['types'] == ['locality', 'political']][0]
    except IndexError:
        return -1

def get_sublocality(result):
    ''' only used for NYDaily news '''
    try:
        return [part['short_name'] for part in result['address_components'] 
                    if part['types'] == ['political', 'sublocality', 'sublocality_level_1']][0]
    except IndexError:
        return -1
    
def get_county(result):
    try:
        return [part['short_name'] for part in result['address_components']
                if part['types'] == ['administrative_area_level_2', 'political']][0]
    except IndexError:
        return -1
     
def periods2bool(periods, bool_fct, ratio=True, input_url=False):
    '''dict of periods, fct ->  applies fct to each article in each period '''
    return np.array([[date, articles2bool(articles, bool_fct, ratio=ratio, input_url=input_url)] for date, articles in periods.items() if len(articles) > 0])

def articles2bool(articles, bool_fct, ratio, input_url):
    results = [bool_fct(article) if not input_url else bool_fct(article, url) for url, article in articles]
    if ratio:
        return sum(results)/len(results) if len(results) != 0 else np.nan
    else:
        return sum(results) if len(results) != 0 else np.nan


def within_city(article, city):
    ''' list of article places -> boolean e.g. LA Weekly: city == Los Angeles'''
    for place in article:
        if get_city(place['result'][0]) == city:
            return True
        else:
            continue
    return False

def within_city_prop(article, city, prop):
    cnt = 0
    for place in article:
        if get_city(place['result'][0]) == city:
            cnt+=1
        else:
            continue
    return cnt/len(article) >= prop if len(article) != 0 else False

def add_ratios(results, ratios=np.arange(.05, 1.05, .05)):
    assert len(results) == len(ratios)
    new_results = []
    for ratio, res in zip(ratios, results):
        new_results.append(np.hstack((res, [[ratio] for _ in range(len(res))])))
    return new_results

def within_dist_prop(article, dist, prop) -> bool:
    cnt = 0
    for place_dist in article:
        if place_dist <= dist:
            cnt+=1
        else:
            continue
    return cnt/len(article) >= prop if len(article) != 0 else False

def within_range(loc1, loc2, max_dist) -> bool:
    return get_dist(loc1, loc2) < max_dist

def get_dist(loc1, loc2):
    return geopy.distance.geodesic(tuple(loc1), tuple(loc2)).km

