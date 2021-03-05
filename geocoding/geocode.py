from itertools import chain
import numpy as np
import re
import time
import pickle
from googlemaps.exceptions import HTTPError

class Geoparse:
    def __init__(self, doc, url, filename, stored_queries, gmap, query=False, save=False):
        self.doc = list(chain(*doc))
        self.filename = filename
        self.entities = self._parse_entities()
        self.url = url.split('/')[2]
        self.queries = stored_queries
        self.gmap = gmap
        self.pub2loc = self._load_pub2_loc()
        self.bounds = {'CA':{'northeast': {'lat': 42.0095169, 'lng': -114.131211},
                             'southwest': {'lat': 32.528832, 'lng': -124.482003}}, 
                       'CO':{'northeast': {'lat': 41.0034439, 'lng': -102.040878},
                             'southwest': {'lat': 36.992424, 'lng': -109.060256}},
                       'MA':{'northeast': {'lat': 42.88679, 'lng': -69.85886099999999},
                             'southwest': {'lat': 41.18705300000001, 'lng': -73.5081419}},
                       'NY':{'northeast': {'lat': 45.102279, 'lng': -70.776005},
                             'southwest': {'lat': 38.947276, 'lng': -79.884686}}}
        if query:
            self.place_ids = self._get_geocodes()
        if save:
            # to be able to create object without writing over a file
            self.file = self.save_places()
        
    def _load_pub2_loc(self):
        with open('pub2loc.p', 'rb') as f:
            pub2loc = pickle.load(f)
        return pub2loc
        
    def _get_state(self, loc):
        return [part['short_name'] for part in loc['address_components'] 
                    if part['types'] == ['administrative_area_level_1', 'political']][0]
        
    def _concat_ents(self, ents):
        if isinstance(ents, dict):
            return ents['text']
        else:
            distance = [int(re.search(r'\d+', ents[i+1]['misc']).group()) 
                        - int(ents[i]['misc'].partition('end_char=')[2]) for i in range(0, len(ents)-1)]
            string = ents[0]['text']
            for space, ent in zip(distance, ents[1:]):
                for i in range(0, space): string+=' '
                string+=ent['text']

            return string
        
    def _parse_entities(self, types='GPE|LOC'):
        entities = [token for token in self.doc 
                          if bool(re.search(types, token['ner']))]
        full_ents, i = [], 0
        while i < len(entities):
            token = entities[i]
            if token['ner'][0] == 'S':
                full_ents.append([token])
            elif token['ner'][0] == 'B':
                ent = [token]
                i += 1
                token = entities[i]
                while token['ner'][0] == 'I':
                    ent.append(token)
                    i+=1
                    token = entities[i]
                if token['ner'][0] == 'E':
                    ent.append(token)
                full_ents.append(ent)
            i+=1
            
        return [self._concat_ents(ent) for ent in full_ents]

    def _get_geocode(self, query, query_gmap):
        ''' '''
        try:
            res = self.queries[query]
        except KeyError:
            if query_gmap:
                bounds = self.bounds[query[1]]
                try:
                    res = self.gmap.geocode(query[0], bounds=bounds)
                except HTTPError:
                    time.sleep(5)
                    try:
                        res = self.gmap.geocode(query[0], bounds=bounds)
                    except:
                        return -1
                if len(res) > 0:
                    self.queries[query] = res
                else:
                    return -1
            else:
                return -1
        
        return res[0]['place_id']

    def _get_geocodes(self):
        area = self._get_state(self.pub2loc[self.url])
        return [self._get_geocode((entity, area), query_gmap=True) for entity in self.entities]

    def save_places(self):
        with open(self.filename, 'w') as f:
            for ID in self.place_ids:
                f.write(str(ID)+'\n')

        return self.filename

    def get_type(self, tag):
        ''' returns named entitites of type tag. e.g. LOC '''
        return [self._parse_entities(types=tag)]

    def get_places(self, tag):
        ''' returns place_ids of entities of type tag (could be multiple tags e.g. LOC|GPE) '''
        area = self._get_state(self.pub2loc[self.url])
        return [self._get_geocode(query=((entity, area)), query_gmap=False) 
                                            for entity in self._parse_entities(types=tag)]
