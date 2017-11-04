# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:26:39 2016

@author: Antonin Bertin & Kevin Serru
"""

# imports
import re, collections
import numpy as np
import difflib
from Levenshtein import ratio


# -  To test KEVIN


def text_to_keywords(words_detected, keywords, keywords_composed, nb_candidates=5):
    """
    Determine the closest keywords to a set of words using Levenshtein ratio.

    Inputs:
    - words_detected = {word}  # detected by OCR in an image
    - keywords = [keyword]  # each keyword is associated with a set of brand_ids / upc
    - keywords_composed = [keyword]  # each keyword is associated with a set of brand_ids (brand with composed names. E.g: Tech Universe)
    - nb_candidates = number of candidate keywords to keep

    Output:
    - candidates = {word: [[keyword, confidence], ...]}  # keywords sorted by decreasing confidence
    - candidates_composed = {word: [[keyword, confidence], ...]}  # keywords sorted by decreasing confidence

    """
    # examples
    # wrong_words = {'cant', 'buter'}
    # wrong_words = {'areone', 'cocon', 'condioner', 'oreal'}
    
    candidates = {}
    candidates_composed = {}
        
    # Words filter policy:
    # - less than 3 characters: ignore
    # - 3 characters: add an ending voyel 'e'
    # - more than 3 characters: compute similarity ratio with list of pre-set keywords
    
    # if all words of length >= 3 are digits, erase word vector
    if all(x.isdigit() for x in words_detected if len(x) >= 3):
        words_detected = []

    for word in words_detected:
        if len(word) < 3:
            continue
        
        else:  
            if len(word) == 3 and not word.isdigit():
                word += 'e'
            
            # Compute distance between word (lowercase) and all keywords then
            # sort by decreasing similarity ratio. Save nb_candidates keywords in kw_candidates.
            # e.g: 
            # word = 'word'
            # ratio = [['world', 80], ['worm', 75], ..., ['potato', 5]] 
            
            keywords_ratio = [[keyword, int(100*ratio(word.lower(), keyword))] for keyword in keywords]
            keywords_ratio.sort(key=lambda x: x[1], reverse=True)
            candidates[word] = keywords_ratio[:nb_candidates]
            
            # Idem with keywords associated with composed brands
            keywords_ratio_composed = [[keyword, int(100*ratio(word.lower(), keyword))] for keyword in keywords_composed]
            keywords_ratio_composed.sort(key=lambda x: x[1], reverse=True)
            candidates_composed[word] = keywords_ratio_composed[:nb_candidates]

    return [candidates, candidates_composed]


# - Ground truth version


# def text_to_keywords(words, keywords, composed_keywords, nb_candidates=3):
#     """
#     Determine the closest keywords to a set of words using Levenshtein ratio.
#     words: SET words that have been detected by the OCR algorithm on an image
#     keywords: LIST of brand/upc keywords
#     nb_candidates: INT number of candidate keywords to keep in kw_candidates
#     kw_candidates: DICT {word:[[keyword0, confidence], [keyword1, confidence], ...]}
#     kw_selected: DICT {word:[keyword0, confidence]}

#     """
#     # examples
#     #wrong_words={'cant','buter'}
#     #wrong_words={'areone','cocon','condioner','oreal'}
    
#     kw_candidates = {}
#     kw_composed_candidates = {}
#     for w in words:
#         if len(w)==3 : w=w+'e'
#         if len(w) >3:
            
#             # normal list
#             ratio_list = [[kw, int(100*ratio(w.lower(), kw))] for kw in keywords]
#             ratio_list.sort(key=lambda x: x[1], reverse=True)  # sort by highest to lowest ratio
#             kw_candidates[w] = ratio_list[:nb_candidates]
            
#             # composed brand list
#             composed_ratio_list = [[ckw, int(100*ratio(w.lower(), ckw))] for ckw in composed_keywords]
#             composed_ratio_list.sort(key=lambda x: x[1], reverse=True)  # sort by highest to lowest ratio
#             kw_composed_candidates[w] = composed_ratio_list[:nb_candidates]
  
#     return [kw_candidates, kw_composed_candidates]
    

def words(text): return re.findall('[a-z0-9\.]+', text.lower()) 

def train(features):
    model = collections.defaultdict(lambda: 1) #take into account unseen words in probability vocbulary distribution
    for f in features:
        model[f] += 1
    return model
    

def edits1(word): # create distance from initial wor of 1 change

   alphabet = 'abcdefghijklmnopqrstuvwxyz'
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes_1    = [a + b[1:] for a, b in splits if b]
   #transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces_1   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in alphabet]


   return set(inserts + deletes_1+ replaces_1 )

def edits2(word): # create distance from initial wor of 1 change

   alphabet = 'abcdefghijlmnoprstuv' #lighter alphabet for deep search   
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   #inserts    = [a + c + b     for a, b in splits for c in alphabet]
   replaces_1   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   #print(inserts+)replaces_1 
   #return set(inserts+replaces_1 )
   return set(replaces_1 )


def known_edits1(words,NWORDS): # create distance from initial word of 2 
    return set(w for w in edits1(words) if w in NWORDS)

def known_edits2(word,NWORDS): # create distance from initial word of 2 
    return set(e2 for e1 in edits2(word) for e2 in edits2(e1) if e2 in NWORDS)

def known_edits_for2(word,NWORDS): # create distance from initial word of 3 
    return set(e3 for e1 in edits2(word) for e2 in edits2(e1)  for e3 in edits2(e2) if e3 in NWORDS)
   
def known(words,NWORDS): return set(w for w in words if w in NWORDS)

def correct(word,NWORDS):


    candidates_1 = known([word],NWORDS) 

    candidates_2 = known_edits1(word,NWORDS)
    
    #candidates_3 = known_edits2(word,NWORDS)
    
    #candidates_3=known_edits_for2(word,NWORDS) # candidates can be used to find 3-distance from initial word; it has been commented due to restriction on computational time    
    candidates = candidates_1 | candidates_2 

    #candidates = candidates_1
    
    if not candidates:

        maxx=word
        short_list=[word]
        confidance=[0] 
    else :    
              
        short_list=list(candidates)
        confidance=np.zeros(len(short_list)) # confidance is the confidance in the result of the matching
        
        for e1 in candidates:
            confidance[short_list.index(e1)]=difflib.SequenceMatcher(None, e1,word).ratio()

        maxx=short_list[np.argmax(np.array(confidance))]


    return [maxx,short_list,confidance]
    
    
def train(features):
    model = collections.defaultdict(lambda: 1) # take into account unseen words in probability vocabulary distribution
    for f in features:
        model[f] += 1
    return model
    

def edits1(word):
    """
    Create distance from initial word of 1 change
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes_1    = [a + b[1:] for a, b in splits if b]
    #transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    replaces_1   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts    = [a + c + b for a, b in splits for c in alphabet]
    return set(inserts + deletes_1+ replaces_1)


def edits2(word):
    """
    Create distance from initial word of 1 change
    """
    alphabet = 'abcdefghijlmnoprstuv' #lighter alphabet for deep search   
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    #inserts    = [a + c + b     for a, b in splits for c in alphabet]
    replaces_1   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    #print(inserts+)replaces_1 
    #return set(inserts+replaces_1 )
    return set(replaces_1 )


def known_edits1(words, NWORDS):
    """
    Create distance from initial word of 2 
    """
    return set(w for w in edits1(words) if w in NWORDS)


def known_edits2(word, NWORDS):
    """
    Create distance from initial word of 2 
    """
    return set(e2 for e1 in edits2(word) for e2 in edits2(e1) if e2 in NWORDS)


def known_edits_for2(word, NWORDS):
    """
    Create distance from initial word of 3 
    """
    return set(e3 for e1 in edits2(word) for e2 in edits2(e1)  for e3 in edits2(e2) if e3 in NWORDS)
   

def known(words, NWORDS): 
    return set(w for w in words if w in NWORDS)


def correct(word, NWORDS):
    candidates_1 = known([word],NWORDS) 
    candidates_2 = known_edits1(word,NWORDS)
    #candidates_3 = known_edits2(word,NWORDS)
    #candidates_3 = known_edits_for2(word,NWORDS) # candidates can be used to find 3-distance from initial word; it has been commented due to restriction on computational time    
    candidates = candidates_1 | candidates_2 
    #candidates = candidates_1
    
    if not candidates:
        maxx = word
        short_list = [word]
        confidance = [0] 
    
    else:          
        short_list = list(candidates)
        confidance = np.zeros(len(short_list)) # confidance is the confidance in the result of the matching
        
        for e1 in candidates:
            confidance[short_list.index(e1)] = difflib.SequenceMatcher(None, e1, word).ratio()

        maxx = short_list[np.argmax(np.array(confidance))]

    return [maxx, short_list, confidance]
