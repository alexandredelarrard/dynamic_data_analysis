# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:30:49 2018

@author: User
"""
import pandas as pd
import numpy as np
from dateutil import relativedelta
from datetime import timedelta


def deduce_match_date(x):
    """
    - tourney_date
    - tourney_end_date 
    - match_num
    - draw_size
    - round
    """    
    
    if x["round"] == "F":
        return x["tourney_end_date"]
    
    if x["draw_size"] <= 64:
        if x["round"] == "SF":
            return x["tourney_end_date"] - timedelta(days=1)
        
        if x["round"] == "QF":
            return x["tourney_end_date"] - timedelta(days=2)
        
        if x["round"] == "R16":
            return x["tourney_end_date"] - timedelta(days=3)
        
        if x["round"] in ["R32", "RR", "BR"]:
                if x["match_num"] in range(x["draw_size"] - 23,x["draw_size"] - 15):
                    return x["tourney_end_date"] - timedelta(days=4)
                else:
                    return x["tourney_end_date"] - timedelta(days=5)
            
        if x["round"] == "R64":

            total_days = (x["tourney_end_date"] - x["tourney_date"]).days - 5
            
            if total_days ==1:
                return x["tourney_end_date"] - timedelta(days=6)
            
            if total_days ==2:
                if x["match_num"] in range(1,17):
                    return x["tourney_end_date"] - timedelta(days=7)
                else:
                    return x["tourney_end_date"] - timedelta(days=6)
            
            if total_days ==3:
                if x["match_num"] in range(1,13):
                    return x["tourney_end_date"] - timedelta(days=8)
                elif x["match_num"] in range(13,25):
                    return x["tourney_end_date"] - timedelta(days=7)
                else:
                    return x["tourney_end_date"] - timedelta(days=6)
                
            if total_days ==4:
                if x["match_num"] in range(1,9):
                    return min(x["tourney_date"], x["tourney_end_date"] - timedelta(days=5))
                elif x["match_num"] in range(9,17):
                    return min(x["tourney_date"]+ timedelta(days=1), x["tourney_end_date"] - timedelta(days=5))  
                elif x["match_num"] in range(17,25):
                    return min(x["tourney_date"]+ timedelta(days=2), x["tourney_end_date"] - timedelta(days=5))  
                else:
                    return min(x["tourney_date"]+ timedelta(days=3), x["tourney_end_date"] - timedelta(days=5))  
                
            else:
                return x["tourney_date"]
                
    elif x["draw_size"] <= 128:
        if x["round"] == "SF":
            return x["tourney_end_date"] - timedelta(days=2)
        
        elif x["round"] == "QF":
            if x["match_num"] in range(x["draw_size"] - 5, x["draw_size"] - 2):
                return x["tourney_end_date"] - timedelta(days=3)
            else:
                return x["tourney_end_date"] - timedelta(days=4)
        
        elif x["round"] == "R16":
            return x["tourney_end_date"] - timedelta(days=5)
            
        
        elif x["round"] == "R32":
            if x["match_num"]  in range(x["draw_size"] - 23, x["draw_size"] - 14):
                return x["tourney_end_date"] - timedelta(days=6)
            else:
                return x["tourney_end_date"] - timedelta(days=7)
            
        elif x["round"] == "R64":
            if x["match_num"] in range(x["draw_size"] - 48, x["draw_size"] - 29):
                return x["tourney_end_date"] - timedelta(days=8)
            else:
                return x["tourney_end_date"] - timedelta(days=9)
            
        elif x["round"] in ["R128", "R96"]:
            if x["match_num"] in range(x["draw_size"] - 80, x["draw_size"] - 61):
                return x["tourney_end_date"] - timedelta(days=10)
            elif x["match_num"] in range(x["draw_size"] - 101, x["draw_size"] - 80):
                return x["tourney_end_date"] - timedelta(days=11)
            elif x["match_num"] in range(x["draw_size"] - 127, x["draw_size"] - 101):
                return x["tourney_end_date"] - timedelta(days=12)
    
    return x["tourney_date"]