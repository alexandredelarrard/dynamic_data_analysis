# -*- coding: utf-8 -*-
"""
Created on Fri May 11 14:50:01 2018

@author: JARD
"""

import os
import warnings
warnings.filterwarnings("ignore")

from create_data.main_create import main_creation
from create_models.main_modelling import main_modelling
from create_finance.main_finance import main_finance

os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"

if __name__ == "__main__":
    
    rebuild = {"redo_missing_atp_statistics" : False,
               "create_elo" : False,
               "create_variable" : True,
               "create_statistics": True
              }    
    
    full_data = main_creation(rebuild=rebuild)
    
    params = {
            "date_test_start" : "2017-01-01", 
            "date_test_end":"2017-12-31"
             }
    
    clf, var_imp, modelling_data = main_modelling(params)

#[Brisbane] : [AUC] 0.9409722222222222 / [Accuracy] 0.9166666666666666 / [logloss] 0.3513542515380929 /  [Match Nbr] 48
#[Doha] : [AUC] 0.9304733727810651 / [Accuracy] 0.8269230769230769 / [logloss] 0.33255091717448804 /  [Match Nbr] 52
#[Chennai] : [AUC] 0.9256198347107438 / [Accuracy] 0.8636363636363636 / [logloss] 0.34996454358439555 /  [Match Nbr] 44
#[Sydney] : [AUC] 0.8442906574394464 / [Accuracy] 0.7647058823529411 / [logloss] 0.4723689006641507 /  [Match Nbr] 34
#[Auckland] : [AUC] 0.8934240362811792 / [Accuracy] 0.7619047619047619 / [logloss] 0.4264464453001329 /  [Match Nbr] 42
#[Australian Open] : [AUC] 0.940688775510204 / [Accuracy] 0.875 / [logloss] 0.3126092888725062 /  [Match Nbr] 224
#[Montpellier] : [AUC] 0.717013888888889 / [Accuracy] 0.6875 / [logloss] 0.6487539547961205 /  [Match Nbr] 48
#[Quito] : [AUC] 0.8515625 / [Accuracy] 0.8125 / [logloss] 0.46395437081810087 /  [Match Nbr] 32
#[Sofia] : [AUC] 0.9421487603305786 / [Accuracy] 0.9318181818181818 / [logloss] 0.310743369881741 /  [Match Nbr] 44
#[Rotterdam] : [AUC] 0.845422116527943 / [Accuracy] 0.7931034482758621 / [logloss] 0.48868877924015297 /  [Match Nbr] 58
#[Buenos Aires] : [AUC] 0.9710743801652892 / [Accuracy] 0.8863636363636364 / [logloss] 0.26004134749316354 /  [Match Nbr] 44
#[Memphis] : [AUC] 0.8335999999999999 / [Accuracy] 0.7 / [logloss] 0.511075294627808 /  [Match Nbr] 50
#[Rio de Janeiro] : [AUC] 0.9281663516068053 / [Accuracy] 0.8043478260869565 / [logloss] 0.343283806482087 /  [Match Nbr] 46
#[Marseille] : [AUC] 0.859375 / [Accuracy] 0.7708333333333334 / [logloss] 0.47518912019828957 /  [Match Nbr] 48
#[Delray Beach] : [AUC] 0.8846153846153846 / [Accuracy] 0.7884615384615384 / [logloss] 0.4252742379467236 /  [Match Nbr] 52
#[Acapulco] : [AUC] 0.7860082304526748 / [Accuracy] 0.6666666666666666 / [logloss] 0.6280774109893374 /  [Match Nbr] 54
#[Sao Paulo] : [AUC] 0.9424 / [Accuracy] 0.88 / [logloss] 0.3306710136681795 /  [Match Nbr] 50
#[Dubai] : [AUC] 0.8622448979591837 / [Accuracy] 0.7678571428571429 / [logloss] 0.4531690981738003 /  [Match Nbr] 56
#[ATP World Tour Masters 1000 Indian Wells] : [AUC] 0.8580987249084712 / [Accuracy] 0.7865168539325843 / [logloss] 0.4724311634382819 /  [Match Nbr] 178
#[ATP World Tour Masters 1000 Miami] : [AUC] 0.859762396694215 / [Accuracy] 0.7556818181818182 / [logloss] 0.46823043832674477 /  [Match Nbr] 176
#[Houston] : [AUC] 0.6597222222222222 / [Accuracy] 0.6458333333333334 / [logloss] 0.705784964064757 /  [Match Nbr] 48
#[Marrakech] : [AUC] 0.9002770083102494 / [Accuracy] 0.7894736842105263 / [logloss] 0.4145853176320854 /  [Match Nbr] 38
#[ATP World Tour Masters 1000 Monte Carlo] : [AUC] 0.9842249657064471 / [Accuracy] 0.9259259259259259 / [logloss] 0.19071098712609252 /  [Match Nbr] 108
#[Budapest] : [AUC] 0.9024000000000001 / [Accuracy] 0.84 / [logloss] 0.39308551555732263 /  [Match Nbr] 50
#[Barcelona] : [AUC] 0.8775826446280992 / [Accuracy] 0.7840909090909091 / [logloss] 0.444211489520967 /  [Match Nbr] 88
#[Estoril] : [AUC] 0.8435374149659864 / [Accuracy] 0.8333333333333334 / [logloss] 0.4754095967149451 /  [Match Nbr] 42
#[Istanbul] : [AUC] 0.9194214876033058 / [Accuracy] 0.8863636363636364 / [logloss] 0.37329739247533406 /  [Match Nbr] 44
#[Munich] : [AUC] 0.8843537414965986 / [Accuracy] 0.7619047619047619 / [logloss] 0.44717237495240714 /  [Match Nbr] 42
#[ATP World Tour Masters 1000 Madrid] : [AUC] 0.9444642221431114 / [Accuracy] 0.8584905660377359 / [logloss] 0.32161426538162763 /  [Match Nbr] 106
#[ATP World Tour Masters 1000 Rome] : [AUC] 0.9688581314878894 / [Accuracy] 0.8921568627450981 / [logloss] 0.23572217941503315 /  [Match Nbr] 102
#[Lyon] : [AUC] 0.9184 / [Accuracy] 0.8 / [logloss] 0.3614768611639738 /  [Match Nbr] 50
#[Geneva] : [AUC] 0.8487712665406427 / [Accuracy] 0.8260869565217391 / [logloss] 0.4798761930255948 /  [Match Nbr] 46
#[Roland Garros] : [AUC] 0.9278721904612734 / [Accuracy] 0.8407079646017699 / [logloss] 0.34088421443815187 /  [Match Nbr] 226
#[s-Hertogenbosch] : [AUC] 0.8923611111111112 / [Accuracy] 0.7916666666666666 / [logloss] 0.4093197876051515 /  [Match Nbr] 48
#[Stuttgart] : [AUC] 0.44618055555555564 / [Accuracy] 0.5 / [logloss] 0.9092132163544496 /  [Match Nbr] 48
#[Halle] : [AUC] 0.8200000000000001 / [Accuracy] 0.7833333333333333 / [logloss] 0.5117620590598866 /  [Match Nbr] 60
#[London / Queen's Club] : [AUC] 0.4081632653061224 / [Accuracy] 0.42857142857142855 / [logloss] 1.0227705386600323 /  [Match Nbr] 56
#[Antalya] : [AUC] 0.8580246913580247 / [Accuracy] 0.6944444444444444 / [logloss] 0.4761889815951387 /  [Match Nbr] 36
#[Eastbourne] : [AUC] 0.671875 / [Accuracy] 0.5833333333333334 / [logloss] 0.7083163000643253 /  [Match Nbr] 48
#[Wimbledon] : [AUC] 0.7532194259742445 / [Accuracy] 0.7155963302752294 / [logloss] 0.6434292549164881 /  [Match Nbr] 218
#[Bastad] : [AUC] 1.0 / [Accuracy] 1.0 / [logloss] 0.2123088954249397 /  [Match Nbr] 48
#[Newport] : [AUC] 0.575 / [Accuracy] 0.45 / [logloss] 0.7593306970782578 /  [Match Nbr] 40
#[Umag] : [AUC] 0.921875 / [Accuracy] 0.8333333333333334 / [logloss] 0.41049029398709536 /  [Match Nbr] 48
#[Gstaad] : [AUC] 0.8393194706994329 / [Accuracy] 0.782608695652174 / [logloss] 0.4905005774420241 /  [Match Nbr] 46
#[Atlanta] : [AUC] 0.8435374149659864 / [Accuracy] 0.7380952380952381 / [logloss] 0.47541888311111724 /  [Match Nbr] 42
#[Hamburg] : [AUC] 0.943758573388203 / [Accuracy] 0.8518518518518519 / [logloss] 0.32861832763861726 /  [Match Nbr] 54
#[KitzbÃ¼hel] : [AUC] 0.9501133786848073 / [Accuracy] 0.8809523809523809 / [logloss] 0.3660331428317087 /  [Match Nbr] 42
#[Washington] : [AUC] 0.6457142857142857 / [Accuracy] 0.6571428571428571 / [logloss] 0.736919904127717 /  [Match Nbr] 70
#[Los Cabos] : [AUC] 0.73 / [Accuracy] 0.65 / [logloss] 0.6245540399104357 /  [Match Nbr] 40
#[ATP World Tour Masters 1000 Canada] : [AUC] 0.7496570644718793 / [Accuracy] 0.6574074074074074 / [logloss] 0.6524736330740981 /  [Match Nbr] 108
#[ATP World Tour Masters 1000 Cincinnati] : [AUC] 0.8497686009255964 / [Accuracy] 0.7452830188679245 / [logloss] 0.49124811384673145 /  [Match Nbr] 106
#[Winston-Salem] : [AUC] 0.8951247165532878 / [Accuracy] 0.8214285714285714 / [logloss] 0.425361858725193 /  [Match Nbr] 84
#[US Open] : [AUC] 0.8783773200720495 / [Accuracy] 0.7743362831858407 / [logloss] 0.4404178886579435 /  [Match Nbr] 226
#[Metz] : [AUC] 0.6086956521739131 / [Accuracy] 0.43478260869565216 / [logloss] 0.8001370187038961 /  [Match Nbr] 46
#[St. Petersburg] : [AUC] 0.7996219281663515 / [Accuracy] 0.7391304347826086 / [logloss] 0.541540763705321 /  [Match Nbr] 46
#[Chengdu] : [AUC] 0.55765595463138 / [Accuracy] 0.5 / [logloss] 0.7410074809323186 /  [Match Nbr] 46
#[Shenzhen] : [AUC] 0.927437641723356 / [Accuracy] 0.8571428571428571 / [logloss] 0.3656067220228059 /  [Match Nbr] 42
#[Tokyo] : [AUC] 0.8594674556213018 / [Accuracy] 0.8076923076923077 / [logloss] 0.4611338490906816 /  [Match Nbr] 52
#[Beijing] : [AUC] 0.9143876337693222 / [Accuracy] 0.8103448275862069 / [logloss] 0.37626291271539986 /  [Match Nbr] 58
#[ATP World Tour Masters 1000 Shanghai] : [AUC] 0.9348000000000001 / [Accuracy] 0.9 / [logloss] 0.3437426713574678 /  [Match Nbr] 100
#[Moscow] : [AUC] 0.8698347107438016 / [Accuracy] 0.8181818181818182 / [logloss] 0.4605855360965837 /  [Match Nbr] 44
#[Stockholm] : [AUC] 0.7201646090534979 / [Accuracy] 0.6296296296296297 / [logloss] 0.5979793056569718 /  [Match Nbr] 54
#[Antwerp] : [AUC] 0.615702479338843 / [Accuracy] 0.4772727272727273 / [logloss] 0.7271151308986273 /  [Match Nbr] 44
#[Basel] : [AUC] 0.9464922711058265 / [Accuracy] 0.8793103448275862 / [logloss] 0.33873188351120415 /  [Match Nbr] 58
#[Vienna] : [AUC] 0.7959183673469388 / [Accuracy] 0.8035714285714286 / [logloss] 0.5490037966415652 /  [Match Nbr] 56
#[ATP World Tour Masters 1000 Paris] : [AUC] 0.6710775047258979 / [Accuracy] 0.6413043478260869 / [logloss] 0.7025871114481402 /  [Match Nbr] 92
#[Nitto ATP Finals] : [AUC] 0.9081632653061223 / [Accuracy] 0.8571428571428571 / [logloss] 0.42688870722694056 /  [Match Nbr] 28
#________________________________________
#[AUC avg] 0.8460500297994965 / [Accuracy avg] 0.7741094700260643 / [logloss] 0.46848354077607957 / [Match Nbr total] 4604 