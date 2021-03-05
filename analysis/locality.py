import pickle
import pandas as pd
import os
import numpy as np
from itertools import chain
from tqdm.notebook import tqdm
import googlemaps 
from geocode import Geoparse
from random import sample
from collections import Counter
gmap = False

############################
### preamble

with open('../geocoding/queries.p', 'rb') as f:
    queries = pickle.load(f)

with open('../geocoding/queries.p', 'rb') as f:
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

def filter(result, pub):
    ''' takes in result and filters it '''
    import re
    # remove common NER false positives
    days = r"Mon\.|Tue\.|Wed\.|Thu\.|Fri\.|Sat\.|Sun\."
    days2 = r"^\s?Mon$|^\s?Tue$|^\s?Wed$|^\s?Thu$|^\s?Fri$|^\s?Sat$|^\s?Sun$"
    pols = r'^\s?(R\-)|^\s?(D\-)' # stuff like R-Arizona sometimes gets matched. we want to avoid this.
    if bool(re.search(days, result['entity'])) or bool(re.search(pols, result['entity'])) or bool(re.search(days2, result['entity'])):
        return []
    elif len(result['result']) != 1: 
        return resolve(results=result, pub=pub.split('/')[2], pub2loc=pub2loc)
    else:
        return result 

def articles2places(articles, file2url, id2place, entity_types='GPE|LOC'):
    ''' Input is a list of articles (path to doc file) and returns list of lists containing
        gmap results for each entity in each article. '''
    all_places = []
    for f in tqdm(articles):
        if f[-1] != 'p':
            doc = openDoc(f[:-3]+'p')
        else:
            doc = openDoc(f)
        pub = file2url[f.split('/')[-1].split('.')[0]]
        try:
            geodoc = Geoparse(doc, pub, 'were not writing', 
                          queries, gmap, query=False, save=False)
            places = [filter(id2place[ID], pub=pub) for ID in geodoc.get_places(entity_types) if ID != -1]
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

from datetime import date, datetime, timedelta
def datespan(startDate, endDate, delta=timedelta(days=1)):
    currentDate = startDate
    while currentDate <= endDate:
        yield currentDate
        currentDate += delta

def split_by_date(articles, url2date, days=7):
    dates = [url2date[url] for url, article in articles]
    start, end = min(dates), max(dates)
    periods = {start_day : [] for start_day in datespan(start, end, delta=timedelta(days=1))}
    for url, article in articles:
        periods[url2date[url]].append([url, article])
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i+n]
    return {days[0]:list(chain(*[periods[d] for d in days])) for days in chunks(list(periods.keys()), n=days)} 

def split_by_dates(places, url2date, pre_e, post_b, pre_b=None, post_e=None):    
    if post_e is None:
        post_e = max(url2date.values())
    if pre_b is None:
        pre_b = min(url2date.values())
    pre_places, post_places = [], []
    for url, place in places:
        if (post_e >= url2date[url]) and (url2date[url] >= post_b):
            post_places.append([url, place])
        elif (pre_b <= url2date[url]) and (url2date[url] <= pre_e):
            pre_places.append([url, place])
    return pre_places, post_places

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

def islocal(article, pub, pub2loc):
    ''' takes in list of place objects and returns whether they are all in the 
    state of the publication. '''
    pub = pub.split('/')[2]
    pub_state = get_state(pub2loc[pub])
    for place in article:
        if len(place['result']) > 1:
            place = resolve(place, pub=pub)
        state = get_state(place['result'][0])
        if state == -1 or state != pub_state:
            return False
    return True

def isNonlocal(article, pub, pub2loc):
    ''' takes in list of place objects and returns whether none of them are in the 
    state of the publication. '''
    pub = pub.split('/')[2] # careful about what 'pub' should look like ... http://www.publication.com/article-title
    pub_state = get_state(pub2loc[pub])
    for place in article:
        if len(place['result']) > 1:
            place = resolve(place, pub=pub)
        state = get_state(place['result'][0])
        if state == pub_state:
            if place['result'][0] == pub2loc[pub]:
                continue
            else:
                return False
    return True

def isNonlocal_dist(article, pub, max_dist, pub2loc):
    ''' takes in list of place objects and returns whether non of them 
    are within a certain distance (in kms). '''
    pub = pub.split('/')[2]
    pub_loc = pub2loc[pub]['geometry']['location'].values()
    for place in article:
        if len(place['result']) > 1:
            place = resolve(place, pub=pub)
        place_loc = place['result'][0]['geometry']['location'].values()
        if geopy.distance.geodesic(tuple(pub_loc), tuple(place_loc)).km < max_dist:
            return False  
    return True

############################
### toponym depth

def place2depth(place, pub):
    ''' input place object. returns depth of hierarchy '''
    if len(place['result']) > 1:
        place = resolve(results, pub=pub)
    d = len(place['result'][0]['address_components'])
    if d != 0:
        return d-1
    else:
        return np.nan

def article2depths(places, pub):
    ''' input is list of place objects out is a list of depths '''
    depths = [place2depth(place, pub.split('/')[2]) for place in places]
    depths = np.array(depths)
    return depths[~np.isnan(depths)]

def articles2depths(articles):
    ''' input is a list of lists of place objects. output is a list of list of depths '''
    return [article2depths(places[1], places[0]) for places in articles]

############################
### trees?

def address2tree(address):
    tree = Tree()
    address = list(reversed(address))
    def helper(lst, tree, parent):
        # base case
        if len(lst) == 0:
            return tree
        else:
            node = lst[0]
            #parent_id = np.random.randint(10000000)
            parent_id = parent+'/'+node['long_name']
            tree.create_node(node['long_name'], parent_id, parent=parent, data=1)
            return helper(lst[1:], tree, parent=parent_id)
    while address[0]['types'][0] == 'postal_code_suffix' or address[0]['types'][0] == 'postal_code':
        address = address[1:] 
    #root_id = np.random.randint(10000000)
    root_id = 'root'
    tree.create_node('ROOT', root_id, data=1)
    return helper(lst=address, tree=tree, parent=root_id)

def paths2length(path1, path2):
    i = 0
    for p1, p2 in zip(path1, path2):
        if p1 == p2:
            i+=1
        else:
            return i
    return i

def merge_trees(tree1, tree2):
    ''' merges an address tree (just one path) to an article tree '''
    path2 = tree2.paths_to_leaves()
    assert len(path2) == 1
    path2 = path2[0]
    max_overlap = -1
    best_path = -1
    paths = tree1.paths_to_leaves()
    # iterate over all possible paths
    for i, path1 in enumerate(paths):
        path_length = paths2length(path1, path2)
        if path_length > max_overlap:
            max_overlap=path_length
            best_path = i
    # get node to merge at
    merge_node_id = paths[best_path][max_overlap-1]
    # merge tree
    tree1.merge(nid=merge_node_id, new_tree=tree2.subtree(nid=merge_node_id))
    # increment nodes
    for node in paths[best_path][0:max_overlap]:
        tree1.get_node(node).data += 1
    return tree1

def article2tree(places):
    ''' input places objects and returns tree '''
    tree = address2tree(places[0]['result'][0]['address_components'])
    for place in places[1:]:
        this_tree = address2tree(place['result'][0]['address_components'])
        tree = merge_trees(tree, this_tree)
    return tree
