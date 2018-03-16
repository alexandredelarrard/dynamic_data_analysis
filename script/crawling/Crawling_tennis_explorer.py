# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 16:39:08 2016

@author: alexandre
"""


from bs4 import BeautifulSoup
import csv
import os
import pandas as pd
import numpy as np
from selenium import webdriver
import time
from datetime import date, timedelta as td



def Get_Players_description(directory, url):
    
    driver = webdriver.Firefox()
    driver.delete_all_cookies()
    f = csv.writer(open(directory + "Players\descriptif.csv", 'wb'))
    href= pd.read_csv(directory + "Players\href.csv",sep=",", header=0)

    for item in href["0"].tolist():
        try:
            driver.get(url+ item)
            
            parent = driver.find_element_by_xpath("//table[@class='plDetail']/tbody/tr")
            soup = BeautifulSoup(parent.get_attribute('innerHTML'), 'html.parser')
            divs= soup.findAll('div')
            
            nb_div = 0
            #### player description
            name = soup.findAll('h3')[0].getText()
            if "Country" in divs[nb_div +1].getText():
                country = divs[nb_div+ 1].getText().replace("Country: ", "")
            else:
                country = "-"
                nb_div-=1
            
            if "Height" in divs[nb_div +2].getText():
                height = int(divs[nb_div+ 2].getText().replace("Height / Weight:", "").split("/")[0].replace("cm", ""))
                weight = int(divs[nb_div + 2].getText().replace("Height / Weight:", "").split("/")[1].replace("kg", ""))
            else:
                height = "-"
                weight = "-"
                nb_div-=1
                
            if "Born" in  divs[nb_div+ 3].getText():       
                born = divs[nb_div+3].getText().replace("Born: ", "")
            else:
                born = "-"
                nb_div -=1
            
            if "Current/Highest rank" in divs[nb_div+4].getText():
                current_rank_single = int(float(divs[nb_div+4].getText().replace("Current/Highest rank - singles: ", "").split("/")[0]))
                best_rank_single = int(float(divs[nb_div+4].getText().replace("Current/Highest rank - singles: ", "").split("/")[1]))
            else:
                current_rank_single = "-"
                best_rank_single = "-"
                nb_div -=1
                
            try:
                best_rank_double = int(float(divs[nb_div+5].getText().replace("Current/Highest rank - doubles: ", "").split("/")[1]))
            except Exception:
                best_rank_double = "-"
                pass
                
            try:
                current_rank_double = int(float(divs[nb_div+5].getText().replace("Current/Highest rank - doubles: ", "").split("/")[0]))
            except Exception:
                current_rank_double ="-"
                pass
            
            gender = divs[nb_div+6].getText().replace("Sex: ", "")
            plays = divs[nb_div+7].getText().replace("Plays: ", "")
            
            #### injuries
            parent = driver.find_element_by_xpath("//table[@class='result flags injured']/tbody")
            soup = BeautifulSoup(parent.get_attribute('innerHTML'), 'html.parser')
            trs = soup.findAll("tr")
            
            nb_wounds = len(parent.find_elements_by_tag_name('tr'))
            
            if nb_wounds>1:
                last_wound = trs[0].findAll("td")[2].getText()
                date_last_wound =trs[0].findAll("td")[0].getText()
            else:
                last_wound = "-"   
                date_last_wound = "-"
            
            if nb_wounds>2:
                next_last_wound = trs[1].findAll("td")[2].getText()
                date_next_last_wound =trs[1].findAll("td")[0].getText()
            else:
                next_last_wound = "-"
                date_next_last_wound = "-"
            print([name, country, height, weight, born, current_rank_single, best_rank_single, current_rank_double, best_rank_double, gender, plays, nb_wounds, last_wound, date_last_wound, next_last_wound, date_next_last_wound])    
            f.writerow([name, country, height, weight, born, current_rank_single, best_rank_single, current_rank_double, best_rank_double, gender, plays, nb_wounds, last_wound, date_last_wound, next_last_wound, date_next_last_wound])    
        
        except Exception:
            pass
        f.close()
    

def Get_ATP_ranking(directory, url):
    
    driver = webdriver.Firefox()
    driver.delete_all_cookies()
    driver.get(url + "/ranking/atp-men/")
    
    page = 1
    max_page =2
    players_list = [] 
    players_href= []
    
    date = driver.find_element_by_xpath("//div[@id='center']/h1").text.split("-")[1].lstrip().rstrip()
    
    while page < max_page:
        ### first get all possible pages
        parent = driver.find_element_by_xpath("//tbody[@class='flags']")
        pages_check = driver.find_element_by_xpath("//div[@class='fright navigator']")
        
        get_pages= []
        for page_elmt in pages_check.find_elements_by_tag_name('a'):
            pages = page_elmt.get_attribute('href')
            get_pages.append(int(pages.split("page")[1].replace("=","")))
            
        max_page = np.max(get_pages)
    
        for elmt in parent.find_elements_by_tag_name('tr'):   
           soup = BeautifulSoup(elmt.get_attribute('innerHTML'), 'html.parser')
           atp_rank = int(float(soup.findAll('td')[0].getText()))
           name = soup.findAll('td')[1].getText()
           points= int(elmt.find_elements_by_class_name("long-point")[0].text)
          
           players_href.append(soup.findAll('a',{"href":True})[0]["href"])
           players_list.append([name, atp_rank, points])
           print([date, name, atp_rank, points])
          
        url_page = pages.replace("page=%s"%page, "page=%s"%(page+1))
        page +=1 
        driver.get(url_page)
        time.sleep(2)
      
    driver.close()
    players = pd.DataFrame(players_list)    
    players.to_csv(directory + "/%s.csv"%date, sep=",", index= False)
    
    return players_href, players


######### declare function                               
if __name__ == "__main__":
    
    directory = r"D:\projects\tennis betting\data\players"
    
    if not os.path.exists(directory):
        os.mkdir(directory)    
    
    url = "http://www.tennisexplorer.com"
    
    ##### crawling over liste a voir elements
    players_href, players = players_desc(directory, url)
    
