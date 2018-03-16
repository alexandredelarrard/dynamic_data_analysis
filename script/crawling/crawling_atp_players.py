# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 21:52:28 2018

@author: User
"""

from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException  
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
from random import shuffle
import shutil
import time   
import tqdm
import json
import glob
import multiprocessing


def players_href(liste_players):
    
     url = "http://www.atpworldtour.com/en/players/"
     driver = webdriver.Firefox()
     driver.delete_all_cookies()
     driver.get(url)
     
     liste_href = []

     for name in tqdm.tqdm(liste_players):
         try:
             inputElement = driver.find_element_by_id("playerInput")
             inputElement.send_keys(name)   
             inputElement.click()
             
             time.sleep(1.5)
             
             drop_down = driver.find_element_by_id("playerDropdown")
             a = drop_down.find_elements_by_tag_name("li")
             
             for i, elmt in enumerate(a):
                 aref = elmt.find_elements_by_tag_name('a')
                 href = aref[0].get_attribute('href')
                 liste_href.append(href.replace("overview", "player-stats"))
                 
             inputElement = driver.find_element_by_id("playerInput").clear()
             
         except Exception as e:
             
             print("Name %s did not worked"%name)
             print(e)
             pass
         
     return liste_href
 
def get_stats_surface_year(driver, href):

    dico = {}
    # get list of years
    parent =  driver.find_element_by_id("playerMatchFactsFilter").text.split("\n")
    
    liste_years = ["0"] + [x for x in parent if x.isdigit()]
    liste_surface = ['Clay','Grass','Hard','Carpet']
    
    for year in liste_years:
        dico[year] = {}
        driver.get(href + "?year=%s&surfaceType=all"%(year))
        dico[year]["all"] = driver.find_element_by_id("playerMatchFactsContainer").text.split("\n")
        
        if "No Player Stats" not in dico[year]["all"]: 
            for surface in liste_surface:
                ref = href + "?year=%s&surfaceType=%s"%(year, surface)
                driver.get(ref)
                dico[year][surface] = driver.find_element_by_id("playerMatchFactsContainer").text.split("\n")

    return dico
 
    
def get_player_desc(driver, href, player):
    
    overall = {}
    driver.delete_all_cookies()
    driver.get(href)
    
    ### get overall desc
    try:
        overall["global_desc"] = driver.find_element_by_class_name("player-profile-hero-overflow").text.split("\n")
    except Exception as  e:
        print(e)
        print("no desc global for %s"%player)
        pass
    
        ### get desc stats per surface per year
    try:
        overall["specific_stats"]= get_stats_surface_year(driver, href)
    except Exception as  e:
        print(e)
        print("no stats for %s"%player)
        pass
    
        ### get bio
    try:
        driver.get(href.replace("player-stats", "bio").replace("overview", "bio"))
        overall["bio"] = driver.find_element_by_id("currentTabContent").text.split("\n")
    except Exception as e:
        print(e)
        print("no bio for %s"%player)
        pass

    return overall


def parse_href(liste_href, path_save_players):
    
    driver = webdriver.Firefox()
    driver.delete_all_cookies()
    
    list_already_done = glob.glob(path_save_players + "/*.json")
    list_already_done = [x.replace(path_save_players, "").replace(".json", "").replace("\\", "") for x in list_already_done]
    
    for href in liste_href:
        player = href.replace("http://www.atpworldtour.com/en/players/", "").split("/")[0]
        print(player)
        
        if not player in list_already_done:
            overall = get_player_desc(driver, href, player)
            
            with open(path_save_players + "/%s.json"%player, "w") as f:
                f.write(json.dumps(overall))
        else:
            print("player %s already done"%player)
    return overall
    
    

def get_list_players(path):
    
    data= pd.read_csv(path)
    
    data["Winner"] = data["Winner"].apply(lambda x : x.lstrip().rstrip().split(" ")[0])
    data["Loser"] = data["Loser"].apply(lambda x : x.lstrip().rstrip().split(" ")[0])
    
    players= set(list(data["Winner"]) + (list(data["Loser"])))
    
    return list(players)


def multiprocess_crawling(liste_href):
    
    jobs = []
    nbr_core = 7
    path_save_players = r"D:\projects\tennis betting\data\players\brute_info"
    
    for i in range(nbr_core):
        
        sub_liste_refs = liste_href[int(i*len(liste_href)/nbr_core): int((i+1)*len(liste_href)/nbr_core)]
        if i == nbr_core -1:
            sub_liste_refs = liste_href[int(i*len(liste_href)/nbr_core): ]
            
        p = multiprocessing.Process(target=parse_href, args=(sub_liste_refs,path_save_players,))
        jobs.append(p)
        p.start()
        
if __name__ == "__main__":
    
    path = r"D:\projects\tennis betting\data\historical\merged.csv"
    path_save_players = r"D:\projects\tennis betting\data\players\brute_info"
    
#    liste_players = get_list_players(path)
    
    ##### crawling over liste a voir elements
#    liste_href = players_href(liste_players)
#    pd.DataFrame(liste_href).to_csv(path_save_players + "/liste_players_href.csv", index= False)
    
    liste_href = pd.read_csv(path_save_players + "/liste_players_href.csv")
    liste_href = liste_href.iloc[:,0].tolist()
    
    players = multiprocess_crawling(liste_href)

    
    
    