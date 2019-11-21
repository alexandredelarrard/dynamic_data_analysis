import pandas as pd
import numpy as np
import scipy.stats as ss
import re
import pdb
import sklearn
import struct
import matplotlib.pyplot as plt
import radar_utils.pmml as pmml
from functools import reduce


def partial_std(df,rensp,weight='Weight'):
    fn_w = (lambda x: df.groupby(x).size()) if weight is None else lambda x: df.groupby(x)[weight].sum()
    def fn(x):
        w=fn_w(x)
        w=w.div(w.sum())**4
        return np.sqrt(np.average((df.groupby(x)[rensp].mean() - df[rensp].mean()) ** 2, weights=w))
    return df.apply(fn, axis=0).sort_values(ascending=False)


def cramers_corrected_stat(factor1, factor2):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    if factor1.std()==0: return 0
    if factor2.std()==0: return 1
    confusion_matrix = pd.crosstab(factor1, factor2)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))  # , np.sqrt(chi2 / (n*(min(confusion_matrix.shape)-1)))

def rec_choose(df, s_list,cnt, threshold):
    new_list=s_list[0:cnt]+[ s_list[j] for j in range(cnt,len(s_list)) if cramers_corrected_stat(df[s_list[cnt-1]],df[s_list[j]]) < threshold]
    if len(new_list)>cnt:
        return rec_choose(df,new_list,cnt+1,threshold)
    else:
        return new_list



def parse_fac(path_fac,complete=False):
    #path_fac=r'D:\Users\RU19563\Desktop\Locale\2017\06 Rotterdam\02 PA01\01 Risk Premium updated\01 Risk Premium\00_db\Frequency_TPL.fac'
    with open(path_fac,'r') as file:
        fac=file.read()

    parse=re.compile('.*?\n(\d+).*?(?:No\. Levels\s*?Base| 1| 2)\n(.*)\*\*\* Factor Rules Section \*\*\*\n(.*)',flags=re.S)
    
    lev_names=parse.match(fac).group(2)
    lev_def=parse.match(fac).group(3)
    split_names=re.compile('.*?\n\d+\t\d+\n')
    parse_names=re.compile('(.*?)\n(\d+)\t(\d+)')
    splitted=map(lambda x: x.split('\n')[:-1],split_names.split(lev_names)[1:])
    names=list(zip(parse_names.findall(lev_names),list(splitted)))

    find_names=re.compile('(.*?)\n\d+\t(String|Float)\n')
    find_rules=re.compile('(-*\d+)\t{(<=|>=|<|>|=)?\s?(.*?)\s?(?:,\s?(<=|>=|<|>|=)?\s?(.*?))?}')
    rules,rules_name,types=find_names.split(lev_def)[1:][2::3],find_names.split(lev_def)[1:][0::3],find_names.split(lev_def)[1:][1::3]
    rules=[find_rules.findall(r) for r in rules]
    rules=dict(zip([n[0][0] for n in names],zip(rules_name,types,rules)))
    
    if complete:
        return names,rules
    else:
        return names

def parse_bid(path_bid,names):
    file=open(path_bid,'br')
    n,p=file.read(4),file.read(4)
    factor_names=map(lambda x: x[0][0],names)
    dt = np.dtype([(i,'<B') for i in factor_names]+[('Weight','<d'),('Response','<d')])
    n=np.fromfile(file, dtype=dt)
    file.close()
    return pd.DataFrame.from_records(n)

def parse_fac_bid(path_fac,path_bid):
    names=parse_fac(path_fac)
    return parse_bid(path_bid,names)


def write_bid(path,df):
    s = b'\x00\x00\x00\x00\x00\x00\xF0\x3F'
    with open(path + name + '.bid', 'wb') as bid_file:
        bid_file.write(struct.pack('<i', 1))
        bid_file.write(struct.pack('<i', n_factors))
        for x in range(n_factors):
            bid_file.write(b'\x01')
        bid_file.write(s)
        bid_file.write(s)

def copy_fac(fr,to,old_bidname,new_bidname):
    new_fac=open(fr,'r').read().replace(old_bidname,new_bidname)
    with open(to,'w') as new_file:
        new_file.write(new_fac)



def univariate(df,x=None,y=None,w=None,ax1=None,reduce_cols=None,autogroup=True):
    if isinstance(y,list):
        cols=[c for c in [x,*y,w] if c is not None]
    else:
        cols = [c for c in [x, y, w] if c is not None]
    if x is None: x=df.index
    ddf=df[cols]
    if autogroup:
        ddf=df[cols].copy()
        if len(df[x].unique())>150:
            dt = sklearn.tree.DecisionTreeClassifier(max_depth=8, min_impurity_decrease=2e-6)
            X, Y = df[x].fillna(0).values.reshape(-1, 1), df[y].values.reshape(-1, 1)
            dt.fit(X, Y)
            pred = dt.predict_proba(df[x].fillna(0).values.reshape(-1, 1))[:, [1]]
            ddf[x]=pred
    sr1=ddf.groupby(x)[w].sum().div(ddf[w].sum()) if w is not None else ddf.groupby(x)[x].count().div(len(ddf))
    ax1=sr1.plot.bar(align='center',ax=ax1,color='#f4f00e',use_index=False,edgecolor='#909293')
    ymax=sr1.max()*2.5
    ax1.set_ylim([0,ymax])
    ax1.yaxis.grid(False)
    ax1.set_xlabel('')
    xmin,xmax=ax1.get_xlim()
    if y is not None:
        sr2=ddf.groupby(x)[y].mean() if w is None else ddf.groupby(x)[y].aggregate(lambda x: np.average(x, weights=ddf[w].loc[x.index]))
        ax2=ax1.twinx()
        if isinstance(sr2,pd.DataFrame) and reduce_cols is not None:
            col_names,funct=reduce_cols
            sr2=sr2.apply(funct,axis=1)
            if isinstance(sr2,pd.DataFrame):
                sr2.columns=col_names
            else:
                sr2.name=col_names
        ymin,ymax=sr2.min()-sr2.mean()*.4,sr2.max()*1.1
        ax2=sr2.plot(ax=ax2,linestyle='--',marker='o',use_index=False)
        if isinstance(ymin,pd.Series):
            ymin,ymax=min(ymin),max(ymax)
        #pdb.set_trace()
        ax2.set_ylim([ymin,ymax])
        ax2.set_xlim([xmin-.5,xmax+.5])
        ax1.set_yticklabels([])
        ax2.yaxis.tick_left()
    ax1.set_title(x if isinstance(x,str) else x.name)
    each=1 if len(sr1.index)<= 11 else 2 if len(sr1.index)<20 else 4
    ax1.set_xticklabels([ x if n % each ==0 else '' for n,x in enumerate(sr1.index)])
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    return (ax1,ax2)


def to_bid(path,df):
    N,p=df.shape
    rfs=p-2
    with open(path,'wb') as f:
        f.write(struct.pack('<ll',N,rfs))
        cast={c:'<B' if i<rfs else '<d' for i,c in enumerate(df.columns)}
        df.astype(cast).to_records(index=False).tofile(f)


def predict_from_beta_export(path,df):
    glm_rels,base=pmml.get_data_xlsx(path)
    return predict_from_diz(df,glm_rels,base)

def predict_from_diz(df,glm_rels,base):
    def tomap(x):
        if isinstance(x.index,pd.MultiIndex):
            relevels=[list(range(1,len(x.index.levels[l])+1)) for l in range(len(x.index.levels))]
            [x.index.set_levels(rel,level=n,inplace=True) for n,rel in enumerate(relevels)]
            x.index=x.index.to_frame().astype(str).apply('_x_'.join,axis=1).values
        else:
            x.index=[i for i in range(1,len(x.index)+1)]
        return x.iloc[:,0].to_dict()
    glm_rels_={tuple(k.split('_x_')):tomap(v) for k,v in glm_rels.items()}

    rf_in_glm=[c[0] for c in glm_rels_.keys() if len(c)==1]
    df_rels=df[rf_in_glm].apply(lambda x: x.map(glm_rels_.get((x.name,),lambda x:0)))
    interaction_in_glm=[c for c in glm_rels_.keys() if len(c)>1]
    mapped=[reduce(lambda x,y: x[1]+'_x_'+y[1],df[list(x)].astype(str).iteritems()).map(glm_rels_.get(x,lambda x:0)) for x in interaction_in_glm]
    df_rels=pd.concat([df_rels,*mapped],axis=1,ignore_index=True)
    df_rels.columns=list(glm_rels.keys())
    df_rels['base']=base
    return df_rels


def get_emblem_model(path):
    import win32com.client  
    EMB=win32com.client.Dispatch("EmbPredict4X.EMBPredict4")
    EMB.LoadModel(path)
    return EMB