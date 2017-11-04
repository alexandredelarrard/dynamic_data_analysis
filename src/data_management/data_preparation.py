# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 05:43:41 2017

@author: alexandre
"""

import pandas as pd
import numpy as np
from sys import stdout
from datetime import datetime

"""
Done :
    - Transform categorical variables into dummies (pre gathering of tooo many categories)
    - Dates transform to the right format when it is possible
    - Nans filled with median or most occuring modality 
"""

"""
To do: 
    - gather modalities of variable depending on the maximum number of modalities (KNN, tree based algo, etc.)
    - FillNans with clustering, prediction method based on correlated variables
    - detect and suppress outliers
    - 
    
"""

def dummify_categorical(data, test = None):
        
    data = data.copy()
    if test:
        test_columns = test.columns
    
    """ suppression of variables with one unique modality """
    colonnes = [x for x in data.columns if len(data[x].value_counts()) ==1]
    data = data.drop(colonnes, axis=1)
    
    if test:
        test = test.drop(colonnes, axis=1)
    
    ListDf = []
    shape_row = data.shape
    
    if test:
        shape_row_test = test.shape
        ListDf_test = []
        
    #### binarize all categorical variables by modality
    for i in data.loc[:, data.dtypes.values == 'O'].columns:
        print(i)
        
        ### if 2 values then 1 variable with 2 real modalities
        if data[i].value_counts().shape[0] == 2:
             YesNo= {data[i].value_counts().index[0]:0 ,data[i].value_counts().index[1] :1}
             data[i] = data[i].apply(YesNo.get).astype(int) 
             
             if test:
                 if i in test_columns:
                     test[i] = test[i].apply(YesNo.get).astype(int) 
                     
        elif data[i].value_counts().shape[0] >= 0.9*len(data[i]):
            del data[i]
            if test and i in test_columns:
                del test[i]
    
        #### otherwise binarized and gather of modalities based on an output 
        elif data[i].value_counts().shape[0] <=10:
            ListDf.append(pd.get_dummies(data[i], prefix=i+'_Mod'))
            if test:
                if i in test_columns:
                    ListDf_test.append(pd.get_dummies(test[i],prefix=i+'_Mod'))
    
    if np.any(data.dtypes == "O"): 
        data = pd.concat([data.loc[:, data.dtypes.values!='O'] , pd.concat(ListDf, axis=1) ], axis=1) 

    print("_ "*50)
    print("train shape from %s, to %s"%(str(shape_row), str(data.shape)))
    
    if test:
        test = pd.concat([test.loc[:,test.dtypes.values!='O'] , pd.concat(ListDf_test, axis=1) ], axis=1)
        print("_ "*50)
        print("test shape from %s, to %s"%(str(shape_row_test), str(test.shape)))
         
        return data, test
     
    return data

### Date gestion with a search for the right format
def date_handling(data):
    
    formats = ('%d/%m/%Y %H:%M:%S AM +00:00', '%d/%m/%Y %H:%M:%S +00:00', '%Y-%m-%d', '%d.%m.%Y', '%d/%m/%y', 
               '%d/%m/%Y','%d%M%Y', "%d%b%Y:%H:%M:%S.%f", "%d%b%Y", "%Y%b%d")
    handled = []
    
    for variable in data.columns:
        if variable not in data.columns:
            print("variable not in dataset")
            return 
        
        def try_parsing_date(data, i):
                
            for fmt in formats:
                try:
                    datetime.strptime(data[i][0], fmt)
                    gfmt = fmt
                    return gfmt
                except ValueError:
                    pass
            return "Error"
        
        if "DATE" in variable.upper() or "TIME" in variable.upper():
            
            
            try:
                form = try_parsing_date(data, variable)
                print(form)
                
                if form != "Error":
                    data[variable] = pd.to_datetime(data[variable], format = form)
                    data[variable+ "year"]   =data[variable].dt.year
                    data[variable + "month"] =data[variable].dt.month
                    data[variable + "month_day"] =data[variable].dt.day
                    data[variable + "weak_day"] =data[variable].dt.dayofweek
                    data[variable+ "year_day"] = data[variable].dt.dayofyear
                    
                    print("variable %s transformed into date with format %s"%(variable, form))
                    handled.append(variable)
                    
            except Exception:
                print("variable %s not transformed into date"%variable)
    
    print("Date handling : %i variables handled (%s)"%(len(handled), handled))        
    
    return data
         

#### Handle missing values
def missing_value(data, test= None, Y_label= None, date_handling = None):
    
    shape_row = data.shape
    suppressed = []
    modified = []
    
    if Y_label:
        if type(Y_label) == str:
            Y_label = [Y_label]
    
    for i in data.columns:
        
        if date_handling:
            data = date_handling(data, i)
            if test:
                test = date_handling(test, i)
             
        ### Nan gestion
        if np.any(pd.isnull(data[i]))==True:     
            print( 'NAN dans bdd Train :  % 25s, with % 7i MVs (% .3f)'  % (i, np.count_nonzero(pd.isnull(data[i])), np.count_nonzero(pd.isnull(data[i]))*100/float(len(data)) ) )
            if i not in Y_label:
                # 3% missing -> median/occurance
                if (np.count_nonzero(pd.isnull(data[i]))*100/float(len(data))<3): 
                    
                    if data.dtypes[i]!='O':
                        value = data[i].median()
                        data[i].fillna(value, inplace=True)
                        if test:
                            test[i].fillna(value,inplace=True)
                            
                    else: 
                        value = data[i].value_counts().index[0]
                        data[i].fillna(value,inplace=True)
                        if test:
                            test[i].fillna(value,inplace=True)
                            
                    modified.append((i, value))
                    
                #### if almost same amount of values than the number of rows than supress 
                elif(np.count_nonzero(pd.isnull(data[i]))*100/float(len(data)) >= 97):
                     del data[i]
                     suppressed.append(i)
                     
                     if test and i in test.columns:
                         del test[i]
                
                else:
                    if data.dtypes[i]!='O':
                        value = -10*abs(min(data[i]))
                        data[i].fillna(value , inplace=True)
                        if test:
                            test[i].fillna(value, inplace=True)
                            
                    else: 
                        value = "others_%s"%str(i)
                        data[i].fillna(value, inplace=True)
                        if test:
                            test[i].fillna(value, inplace=True)
                            
                    modified.append((i, value))
                      
            ### in case of missing values in test but not in train
            if test:
                if np.any(pd.isnull(test[i]))==True: 
                    if i not in Y_label:
                         print( 'NAN dans bdd Test : % 25s, with % 7i MVs (% .3f)' %(i, np.count_nonzero(pd.isnull(test[i])), np.count_nonzero(pd.isnull(test[i]))*100/float(len(test)) ) )
                    
                         if (np.count_nonzero(pd.isnull(data[i]))*100/float(len(data))<3):
                   
                            if test.dtypes[i]!='O':
                                test[i].fillna(test[i].median(), inplace=True)
                                
                            else: 
                                test[i].fillna(test[i].value_counts().index[0],inplace=True)
                        
                         else:
                            if data.dtypes[i]!='O':
                                test[i].fillna(-10*abs(min(test[i])) , inplace=True)
                                
                            else: 
                                test[i].fillna("others_%s"%str(i), inplace=True)
                                                     
    print("_ "*30)
    
    print("variables suppressed : %s\n"%str(suppressed))
    print("Modified variables with replaced value %s\n"%modified)
    print("data shape from %s, to %s"%(str(shape_row), str(data.shape)))
    
    print("_ "*30)
    
    if test:
        return data, test 
    
    return data

def into_object_variable(data, variables):
    
    for var in variables:
        data[var] = data[var].astype(str)
        
    return data
        
