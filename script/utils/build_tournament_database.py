# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:45:20 2018

@author: User
"""

## data come from https://github.com/JeffSackmann/tennis_atp
## https://github.com/JeffSackmann

import pandas as pd
import numpy as np
import re

parite_euro_currency = {
                    "$":{"2000": 0.923612,
                       "2001": 0.895571,
                       "2002": 0.945574,
                       "2003": 1.131148,
                       "2004": 1.243943,
                       "2005": 1.244114,
                       "2006": 1.255623,
                       "2007": 1.370478,
                       "2008": 1.470755,
                       "2009": 1.394759,
                       "2010": 1.325695,
                       "2011": 1.391930,
                       "2012": 1.284789,
                       "2013": 1.328118,
                       "2014": 1.328501,
                       "2015": 1.109513,
                       "2016": 1.106903,
                       "2017": 1.129686,
                       "2018": 1.226949},
                    "AU$":{"2000": 1.588947,
                       "2001": 1.731890,
                       "2002": 1.737662,
                       "2003": 1.737941,
                       "2004": 1.690515,
                       "2005": 1.631969,
                       "2006": 1.666811,
                       "2007": 1.634836,
                       "2008": 1.741623,
                       "2009": 1.772696,
                       "2010": 1.442176,
                       "2011": 1.348444,
                       "2012": 1.240711,
                       "2013": 1.377695,
                       "2014": 1.471874,
                       "2015": 1.477661,
                       "2016": 1.488282,
                       "2017": 1.473225,
                       "2018": 1.548992},
                    "£":{"2000": 0.609478,
                       "2001": 0.621815,
                       "2002": 0.628828,
                       "2003": 0.691993,
                       "2004": 0.678671,
                       "2005": 0.683785,
                       "2006": 0.681729,
                       "2007": 0.684337,
                       "2008": 0.796285,
                       "2009": 0.890916,
                       "2010": 0.857826,
                       "2011": 0.867893,
                       "2012": 0.810871,
                       "2013": 0.849255,
                       "2014": 0.806120,
                       "2015": 0.725850,
                       "2016": 0.819483,
                       "2017": 0.876679,
                       "2018": 0.883738}
                     }


def homogenize_prizes(x, dico,year):
    regex = re.compile('["€","$", "£", "AU$"]')
    if x[1] == "€":
        x[0] = int(regex.sub("", str(x[0])).replace(",","").replace(" ","").replace("NC", "0"))
    else:
        x[0] = int(regex.sub("", str(x[0])).replace(",","").replace(" ","").replace("NC", "0"))*dico[x[1]][str(year)]
    return  x[0]
    

def read_transform_data(path, year, first=False):
    data = pd.read_csv(path + "/%i.csv"%year, header=None)
    data.rename(columns = {0 : "Date", 1 :"Tournament", 2 :"Type", 3:"Prize", 4 : "Surface"}, inplace=True)
    
    data["Date"] = pd.to_datetime(data["Date"], format = "%d/%m/%Y")

    data["City"] = data["Tournament"].apply(lambda x: x.split(", ")[1].lower())
    data["Indoor_flag"] = (data["Surface"].apply(lambda x: x.split(" ")[1]) == "(int.)")*1
    data["Surface"] = data["Surface"].apply(lambda x: x.split(" ")[0])
        
    data["Tournament"] = data["Tournament"].apply(lambda x: x.split(", ")[0].lower())
    data["Currency"] = data["Prize"].apply(lambda x: [u for u in ["AU$", "€", "$", "£"] if u in x][0])
    data["Prize"] = data[["Prize", "Currency"]].apply(lambda x: homogenize_prizes(x, parite_euro_currency, year), axis=1)

    return data

if __name__ == "__main__":
    path = r"C:\Users\JARD\Documents\Projects\tennis"
    
    for year in range(2000,2019):
        print(year)
    
        data = read_transform_data(path, year, first = True)
        overall_data= data
        ref= data["Tournament"].tolist()
        data["year"] = year
        
        data = data[["Tournament", "City", "Date", "Prize", "Surface", "Indoor_flag", "Currency"]]
        data.to_csv(r"C:\Users\JARD\Documents\Projects\tennis\%i_.csv"%year)


merg = pd.read_csv(r"C:\Users\JARD\Documents\Projects\tennis\merged.csv")

m = merg[["Tournament", "Date"]]
m["Date"] = pd.to_datetime(m["Date"] , format = "%Y-%m-%d") 
m["year"] =m["Date"].dt.year
m["month"] =m["Date"].dt.month
m["day"] =m["Date"].dt.day

m = m.drop_duplicates(["Tournament", "year"])
del m["Date"]

m.to_csv(r"C:\Users\JARD\Documents\Projects\tennis\match_tournaments_V2.csv")