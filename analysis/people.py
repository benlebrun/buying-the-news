import re
import string
from collections import Counter
from itertools import chain
import numpy as np

def _concat_ents(ents):
    if isinstance(ents, dict):
        return ents['text']
    else:
        distance = [int(re.search(r'\d+', ents[i+1]['misc']).group()) 
                        - int(ents[i]['misc'].partition('end_char=')[2]) for i in range(0, len(ents)-1)]
        string = ents[0]['text']
        for space, ent in zip(distance, ents[1:]):
            for _ in range(0, space): string+=' '
            string+=ent['text']
        return string
        
def _parse_entities(doc, types='PERSON', concat=True):
    entities = [token for token in doc if bool(re.search(types, token['ner']))]
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
    return [_concat_ents(ent) if concat else [e['text'].lower() for e in ent] for ent in full_ents]

def is_composite(name):
    ''' whether extracted name is more than just one word '''
    return len(name) > 1

def split_name(name):
    ''' returns (last, first)'''
    name = [n for n in name if n not in ['.', '(', ')']]
    for _ in range(0, name.count('-')):
        # combine words between '-' e.g. (john, this, -, that) ->  (john, this-that)
        i = name.index('-')
        if i == 0:
            name = name[1:]
        elif i == len(name)-1:
            name = name[:-1]
        else:
            name = name[:i-1] + [''.join(name[i-1:i+2])] + name[i+2:]            
    if len(name) > 2:
        extras = []
        while name[-1] in ['jr.', 'jr', 'senior', 'junior', 'dr', 'dr.', 'sr', 'sr.']:
            extras.append(name[-1])
            # avoid common ends
            name = name[:-1]
        return name[-1], name[0]+'-'.join(extras)
    elif len(name) == 2:
        return name[-1], name[0]
    else:
        return name[0], False

def cnts2firstnames(cnts):
    return np.unique(list(chain(*[firsts.keys() for name, firsts in cnts.items()])))

def names2paired(names):
    '''pair single names with a composite one'''
    name_cnts = {}
    for name in names:
        last, other = split_name(name)
        if other:
            '''name is a composite'''
            if last in name_cnts.keys():
                # check if matches exist
                try:
                    name_cnts[last][other] += 1
                except KeyError:
                    name_cnts[last][other] = 1
            else:
                # create an entry
                name_cnts[last] = {other:1}
        else:
            '''check composites for matches'''
            # first check for last name matches
            if last in name_cnts.keys():
                firsts = list(name_cnts[last].keys())
                # if only one last name match exists, pair this name to that one
                if len(firsts) == 1:
                    name_cnts[last][firsts[0]] += 1
                else:
                    # there are multiple people with this last name
                    # there is no way to know which one is which
                    try:
                        name_cnts[last]['NA'] += 1
                    except KeyError:
                        name_cnts[last]['NA'] = 1
            else:
                first_names = cnts2firstnames(name_cnts)
                if last in first_names:
                    if Counter(first_names)[last] == 1:
                        # this name has been used as a first name once
                        # so add a count to it
                        for last_name, first_names in name_cnts.items():
                            if last in first_names:
                                name_cnts[last_name][last] += 1
                                break
                    elif Counter(first_names)[last] == 0:
                        print('error')
                    else:
                        # it has been used as a first name more than once
                        print(last, 'has been used as a first name more than once')
                else:
                    # otherwise it is an unknown name
                    name_cnts[last] = {'NA':1}
    return name_cnts 

def article2names(document):
    return names2paired(_parse_entities(list(chain(*document)), concat=False))

def are_same_level(names):
    ''' whether list or dict of people are same level (state, local, or national)'''
    if isinstance(names, list):
        # specific first name
        levels = [n['level'] for n in names]
        same_level = len(set(levels)) == 1
        return levels[0], same_level
    else:
        # last name
        levels = list(chain(*[[person['level'] for person in people] for first, people in names.items()]))
        same_level = len(set(levels)) == 1
        return levels[0], same_level

def get_years(names):
    if isinstance(names, list):
        # specific first name
        return list(chain(*[n['years'] for n in names]))
    else:
        # last name
        return list(chain(*[list(chain(*[person['years'] for person in people])) for first, people in names.items()]))

def name2match(last, first_n, year, lastnames, printed=[]):
    ''' 
    attempts to pair an extracted name to a name in lastnames
    lastnames avoids reprinting the same errors all the time
    '''
    try:
        result = lastnames[last]
        if first_n == 'NA':
            # are all people from the same group?
            # or is there only one person at this level?
            level, same_level = are_same_level(result)
            if same_level:
                # double check with year
                years = get_years(result)
                if year in years:
                    # if year + last name matches, and there are no conflicts,
                    # match it
                    return level
                else:
                    return False
            else:
                return False
        else:
            try:
                first = result[first_n]
            except KeyError:
                with open('name_match_errors.txt', 'a+') as f:
                    f.write(f'{last},{first_n} ')
                return False
            # are there any conflicts?
            level, same_level = are_same_level(first)
            if same_level:
                for entry in first:
                    #if isinstance(entry['years'], float):
                     #   return level
                    if year in entry['years']:
                        return level
                # no date match - no good
                if f"did not match {first_n} {last} due to year - {level}" not in printed:
                    #print(f"did not match {first_n} {last} due to year - {level}")
                    printed.append(f"did not match {first_n} {last} due to year - {level}")
                return False
            else:
                # try to disambiguate by year
                possible_levels = []
                for entry in first:
                    #if isinstance(entry['years'], float):
                     #   continue
                    if year in entry['years']:
                        possible_levels.append(entry['level'])
                if len(possible_levels) == 1:
                    #print(f"matched {last} to {possible_levels[0]} ({year})")
                    return possible_levels[0]
                else:
                    #print(f"did not match {first_n} {last} ({year})")
                    return False
    except KeyError:
        # last name doesn't match
        return False


    