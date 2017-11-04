# -*- coding: utf-8 -*-

"""
Author: Kevin Serru
Email : kev.serru@gmail.com
Date  : 23/06/2016
"""

# imports
import sys
import os
import pandas as pd
import numpy as np

# Additional imports
root = os.path.dirname(os.path.realpath(__file__))
sys.path.append(root)
import COMMON as shared_paths
root = shared_paths.data_root_path

# -
# NOTE to BENOIT
# If you want slightly faster execution, and you are focusing on only a few number of families,
# you may want to pre-charge all relevant families' csv using the function precharge_family_csv
# Then you can pass a family dataframe as additional argument in the majority of functions
# (who require a family dataframe)


def precharge_family_csv(fam_id_list, general_database_path):
    """
    Returns a list of PANDAS DATAFRAMES associated with a LIST of family ids.
    :param fam_id_list:
    :param general_database_path:
    :return:
    """
    df_list = []
    existing_fam_id = get_families(general_database_path)
    for fam_id in fam_id_list:
        if fam_id in existing_fam_id:
            df = pd.read_csv('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"]), sep=";", dtype={'upc': str, 'brand_id': int})
            df_list.append(df)
        else:
            df_list.append(None)
    return df_list


def get_bricks_from_fam_id(fam_id, path_to_dictionaries):
    """
    Returns all the bricks (names) in a family.
    """
    fam_brick_df = pd.read_csv(path_to_dictionaries+'/family_brick.csv', sep=";", index_col="family_id", dtype={'family_id': int})
    fam_id = int(fam_id)
    if fam_id in fam_brick_df.index.tolist():
        fam_brick = fam_brick_df["brick"][fam_id].tolist()
    else:
        fam_brick = ""
    return fam_brick


def get_fam_name_from_fam_id(fam_id, path_to_dictionaries=root+'/dictionaries'):
    """
    Returns the family name from fam_id
    :param fam_id:  STRING or LIST
    :param general_database_path:  STRING
    :param path_to_dictionaries:  STRING
    :return: STRING family name
    """
    # load dictionary
    families_df = pd.read_csv(path_to_dictionaries+'/families.csv', sep=";", index_col="family_id", dtype={'family_id': int})

    fam_name = None
    if type(fam_id) in [str, int, float]:
        fam_id = int(fam_id)
        if fam_id in families_df.index:
            fam_name = families_df["family"][fam_id]
        else:
            fam_name = None
    elif type(fam_id) == list:
        fam_name_list = []
        for f_id in fam_id:
            f_id = int(f_id)
            if f_id in families_df.index:
                fam_name_list.append(families_df["family"][f_id])
            else:
                fam_name_list.append(None)
        fam_name = fam_name_list

    return fam_name


def get_brand_name_from_brand(brand_id, path_to_dictionaries=root+'/dictionaries'):

    if type(brand_id) == str:
        if '.' in brand_id:
            brand_id = brand_id.split('.', 1)[0]


    # load dictionary
    brands_df = pd.read_csv(path_to_dictionaries+'/brands.csv', sep=";", index_col="brand_id", dtype={'brand_id': int})
    brand_name = None
    if type(brand_id) in [str, int, float]:

        brand_id = int(brand_id)

        if brand_id in brands_df.index.tolist():
            brand_name = brands_df["brand"][brand_id]
        else:
            brand_name = None
    elif type(brand_id) == list:
        brands_name_list = []
        for b_id in brand_id:
            b_id = int(b_id)
            if b_id in brands_df.index.tolist():
                brands_name_list.append(brands_df["brand"][b_id])
            else:
                brands_name_list.append(None)
        brand_name = brands_name_list

    return brand_name


def get_families(general_database_path):
    # formerly get_section
    """
    Returns a list of all families in general_database_storage
    :param general_database_path:  STRING
    :return: LIST of family id's
    """
    families = []
    for d in next(os.walk(general_database_path))[1]:
        if d.isdigit():
            # dir is a family folder
            families.append(d)
    return families

def get_brands_from_family(fam_id, general_database_path, df=None):
    """
    Returns a LIST of brand ids associated with a family id.
    :param fam_id: STRING family id
    :param general_database_path: STRING path to general_database_storage
    :param df: DATAFRAME dataframe associated with family
    :return:
    """
    brands = []
    print '/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"])
    if os.path.isfile('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"])):
        if df is None:
            df = pd.read_csv('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"]), sep=";", dtype={'upc': str, 'brand_id': int})
        brands = list(df.brand_id.unique())
    return brands


def get_index_from_brand(brand_id, fam_id, general_database_path, df=None):
    """
    Returns the brand index relative to family id.
    :param brand_id:
    :param fam_id:
    :param general_database_path:
    :param df: DATAFRAME dataframe associated with family
    :return:
    """
    brand_id = int(brand_id)
    bf_index = None
    if os.path.isfile('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"])):
        if df is None:
            df = pd.read_csv('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"]), sep=";", dtype={'upc': str, 'brand_id': int, 'bf_index': int})
        if brand_id in df['brand_id'].get_values():
            i = df.loc[df['brand_id'] == brand_id]['bf_index'].unique()
            if len(i)!=1:
                print("ERROR in get_index_from_brand. \n"
                      "In family: %s, the index associated with brand: %s is %s "
                      "(should be an array of 1 int)" % (fam_id, str(brand_id), str(i)))
            else:
                bf_index = i[0]
    return bf_index

def get_brand_from_index(bf_index, fam_id, general_database_path, df=None):
    """
    Returns the brand id associated with brand index relative to family id.
    :param bf_index:
    :param fam_id:
    :param general_database_path:
    :param df: DATAFRAME dataframe associated with family
    :return:
    """
    brand_id = None
    bf_index = int(bf_index)
    if os.path.isfile('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"])):
        if df is None:
            df = pd.read_csv('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"]), sep=";", dtype={'bf_index': int,'brand_id': int})

        if bf_index in df['bf_index'].get_values():
            i = df.loc[df['bf_index'] == bf_index]['brand_id'].unique()
            if len(i)!=1:
                print("ERROR in get_brand_from_index. \n"
                      "In family: %s, the brand associated with index: %s is %s "
                      "(should be an array of 1 int)" % (fam_id, str(bf_index), str(i)))
            else:
                brand_id = i[0]
    return brand_id


def get_upc_from_brand(brand_id, fam_id, general_database_path, df=None):
    """
    Returns a list of upcs associated with brand id in the specified family.
    :param brand_id:
    :param fam_id:
    :param general_database_path:
    :param df: DATAFRAME dataframe associated with family
    :return:
    """
    brand_id = int(brand_id)
    upc_list = []
    if os.path.isfile('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"])):
        if df is None:
            fam_df = pd.read_csv('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"]), sep=";", index_col='brand_id', dtype={'upc':str, 'brand_id': int})
        else:
            fam_df = df.copy()
            fam_df.set_index('brand_id', inplace=True)

        if brand_id in fam_df.index:
            a = fam_df['upc'][brand_id]
            if type(a) == str or type(a) == int or type(a) == np.int64: upc_list = [a]
            else: upc_list = a.tolist()

    p = [int(x) for x in upc_list]
    for i, value in enumerate(p):
        if value == 0:
            del upc_list[i]

    final_list = []
    for ul in upc_list:
        if len(str(ul)) < 14:  # add proper number of 0
            u = "".join(["0" for i in range(14-len(str(ul)))]) + str(ul)
            final_list.append(u)
        else:
            final_list.append(str(ul))

    return final_list


def get_brand_from_upc(upc, fam_id, general_database_path, df=None):


    upc = str(upc)
    upc_int = int(upc)
    if len(upc) < 14:  # add proper number of 0
        upc = "".join(["0" for i in range(14-len(upc))]) + upc

    brand_id = None

    if os.path.isfile('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"])):
        if df is None:
            fam_df = pd.read_csv('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"]), sep=";", index_col='upc', dtype={'upc':int, 'brand_id':int})
        else:
            fam_df = df.copy()
            fam_df['upc'] = fam_df['upc'].astype(int)
            fam_df.set_index('upc', inplace=True)

        if upc_int in fam_df.index:
            brand_id = int(fam_df['brand_id'][upc_int])

    return brand_id


def get_index_from_upc(upc, fam_id, general_database_path, df=None):
    """
    /!\ You can pass a single upc or a list of upcs
    Note: no need to enter brand_id as input since a upc is unique
    and cannot be in two brands at once.
    Returns the upc index relative to its brand in the specified family.
    :param upc:
    :param fam_id:
    :param general_database_path:
    :param df:
    :return:
    """
    upcb_index = None
    if os.path.isfile('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"])):
        if df is None:
            df = pd.read_csv('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"]), sep=";", dtype={'upc': str, 'upcb_index':int})
        if type(upc) == list:
            for u in upc:
                if u in df['upc']:
                    i = df.loc[df['upc'] == upc]['upcb_index'].get_values()
                    if len(i)>1:
                        print("ERROR in get_index_from_brand. \n"
                              "In family: %s, the index associated with upc: %s is %s "
                              "(should be an array of 1 int)" % (fam_id, u, str(i)))
                    else:
                        upcb_index = i[0]
        else:
            if upc in df['upc'].get_values():

                i = df.loc[df['upc'] == upc]['upcb_index'].get_values()
                if len(i)>1:
                    print("ERROR in get_index_from_brand. \n"
                          "In family: %s, the index associated with upc: %s is %s "
                          "(should be an array of 1 int)" % (fam_id, upc, str(i)))
                else:
                    upcb_index = i[0]
    return upcb_index


def get_upc_from_index(upcb_index, brand_id, fam_id, general_database_path, df=None):
    """
    Returns the upc associated with upc index relative to the specified brand id in family id.
    :param upcb_index:
    :param brand_id:
    :param fam_id:
    :param general_database_path:
    :param df:
    :return:
    """
    upc = None
    upcb_index = int(upcb_index)
    brand_id = int(brand_id)

    if os.path.isfile('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"])):

        if df is None:
            df = pd.read_csv('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"]), sep=";", dtype={'upc': str, 'brand_id':int, 'upcb_index':int})

        if upcb_index in df.loc[df['brand_id'] == brand_id]['upcb_index'].tolist():
            i = df.loc[df['brand_id'] == brand_id].loc[df['upcb_index'] == upcb_index]['upc'].tolist()

            if len(i)>1:
                print("ERROR in get_index_from_brand. \n"
                      "In family: %s, the upc associated with brand: %s and upcb_index: %s is %s "
                      "(should be an array of 1 int)" % (fam_id, str(brand_id), str(upcb_index), str(i)))
            else:
                upc = i[0]
    return upc





def get_images_from_upc(upc, fam_id, general_database_path, im_database_reference_path, im_database_path, brand_id=None, df=None, include_shelf=True, include_ref=True):
    """
    Returns a dictionary with 2 keys: 'true' and 'shelf // values: [] lists of absolute paths to images
    from im_reference_database and im_database.
    If no image is available for the specified upc, a WARNING messageis prompted and the value list is empty.
    :param upc: STRING
    :param fam_id: STRING
    :param general_database_path:STRING
    :param im_database_reference_path: STRING
    :param im_database_path: STRING
    :param df: PANDAS DATAFRAME
    :param include_shelf: BOOLEAN
    :param shelf_only: BOOLEAN
    :return: DICT
    """

    if brand_id:
        brand_id = int(brand_id)

    upc = str(upc)

    if upc != "no_upc":
        upc_int = int(upc)
        if len(upc) < 14:  # add proper number of 0
            upc = "".join(["0" for i in range(14-len(upc))]) + upc

    images = {'true': [], 'shelf': []}

    if brand_id is None:
        if os.path.isfile('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"])):
            if df is None:
                fam_df = pd.read_csv('/'.join([general_database_path, fam_id, "database_"+fam_id+".csv"]), sep=";", index_col='upc', dtype={'upc':int, 'brand_id':int})
            else:
                fam_df = df.copy()
                fam_df['upc'] = fam_df['upc'].astype(int)
                fam_df.set_index('upc', inplace=True)

            if upc_int in fam_df.index:
                brand_id = int(fam_df['brand_id'][upc_int])
                print(brand_id)

    if brand_id is not None:
        if include_ref and upc != "no_upc":
            # include im from im_database_reference
            im_true = []
            if os.path.isdir('/'.join([im_database_reference_path, fam_id, "images", str(brand_id), upc])):
                path_to_upc = '/'.join([im_database_reference_path, fam_id, "images", str(brand_id), upc])
                im_true = ['/'.join([path_to_upc, upc_image_file]) for upc_image_file in next(os.walk(path_to_upc))[2]]

            if len(im_true) == 0:
                print("no im_database_reference images for upc: %s" % upc)
            else:
                images['true'] = im_true

        if include_shelf:
            # include im from im_database (from shelf segmentation)
            im_shelf = []
            if os.path.isdir('/'.join([im_database_path, fam_id, "images", str(brand_id), upc])):
                path_to_upc = '/'.join([im_database_path, fam_id, "images", str(brand_id), upc])
                im_shelf = ['/'.join([path_to_upc, upc_image_file]) for upc_image_file in next(os.walk(path_to_upc))[2]]

            if len(im_shelf) == 0:
                print("no im_database images for upc: %s" % upc)
            else:
                images['shelf'] = im_shelf

    return images


def get_shelf_images_and_xlsx(fam_id, im_database_shelf_path):
    """
    Returns a dictionary with 2 keys: 'xlsx' and 'images' // values: [] absolute path to xlsx or images.
    :param fam_id: STRING
    :param im_database_shelf_path: STRING
    :return: DICT
    """
    shelf = {'xlsx': [], 'images': []}
    path_to_xlsx = '/'.join([im_database_shelf_path, fam_id, "data"])
    path_to_images = '/'.join([im_database_shelf_path, fam_id, "images"])
    if len(next(os.walk(path_to_xlsx))[2]) == 0 or len(next(os.walk(path_to_images))[2]) == 0:
        print("WARNING: in get_shelf_images_and_xlsx. No shelf images associated with family: %s" % fam_id)
    else:
        shelf['xlsx'] = ['/'.join([im_database_shelf_path, fam_id, "data", x]) for x in next(os.walk(path_to_xlsx))[2]]
        shelf['images'] = ['/'.join([im_database_shelf_path, fam_id, "images", x]) for x in next(os.walk(path_to_images))[2]]
    return shelf


def get_brands_from_keyword(keyword, fam_id, general_database_storage, keywords_fam_df=None):
    """
    Returns a string that is a list of brands associated with a keyword.
    :param keyword:
    :param fam_id:
    :param general_database_storage:
    :param keywords_fam_df:
    :return:
    """
    brands = None
    # check existence of fam keywords csv file
    if keywords_fam_df is None:
        if os.path.isfile('/'.join([general_database_storage, fam_id, "keywords_%s.csv" % fam_id])):
            keywords_fam_df = pd.read_csv('/'.join([general_database_storage, fam_id, "keywords_%s.csv" % fam_id]), sep=";", index_col='word')
            if keyword in keywords_fam_df.index.tolist():
                brands = keywords_fam_df['brands'][keyword]
    else:
        keywords_fam_df.set_index('word', inplace=True)
        if keyword in keywords_fam_df.index.tolist():
            brands = keywords_fam_df['brands'][keyword]

    return brands

#
# --- LOGITECH --- #
#

def get_price_from_upc(upc, folder="logitech", qopius_storage_path=root, price_range_df=None):
    upc = int(upc)
    price_range = ""
    # check existence of price_range csv file
    if price_range_df is None:
        if os.path.isfile('/'.join([qopius_storage_path, folder, "price_range.csv"])):
            price_range_df = pd.read_csv('/'.join([qopius_storage_path, folder, "price_range.csv"]), sep=";", index_col='upc', dtype={'upc':int})

            if upc in price_range_df.index:
                price_range = price_range_df['price_range'][upc]
    else:
        price_range_df.set_index('upc', inplace=True)
        if upc in price_range_df.index:
            price_range = price_range_df['price_range'][upc]

    return price_range


def keyword_in_name(upc, brand_id, fam_id, keyword, im_database_reference=root+'/im_database_reference', df=None):
    name = ""
    upc = int(upc)
    if df is None:
        if os.path.isfile('/'.join([im_database_reference, fam_id, "data", "%s.csv" % str(brand_id)])):
            df = pd.read_csv('/'.join([im_database_reference, fam_id, "data", "%s.csv" % str(brand_id)]), sep=";", index_col='upc', dtype={'upc':int})
            if upc in df.index:
                name = df['name'][upc]
    else:
        df.set_index('upc', inplace=True)
        if upc in df.index:
            name = df['name'][upc]

    if keyword in list(map(lambda x:x.lower(), name.split())):
        return True
    else:
        return False

#
# --- Fin - LOGITECH --- #
#


def get_brand_name_from_index(bf_index, fam_id, general_database_path, path_to_dictionaries, df=None):
    """
    Returns the brand name associated with brand index relative to family id.
    Of bf_index is a list them the function returns a list of brand names
    :param bf_index: int/str or list of int/str
    :param fam_id: int/str
    :param general_database_path: str
    :param df: DATAFRAME dataframe associated with family
    :return: str or list of str
    """
    brand_name = None
    if type(bf_index) == str or type(bf_index) == int:
        brand_id = get_brand_from_index(bf_index, str(fam_id), general_database_path, df)
        brands_df = pd.read_csv(path_to_dictionaries+'/brands.csv', sep=";", index_col='brand_id', dtype={'brand_id':int, 'brand':str})
        if brand_id in brands_df.index:
            brand_name = brands_df['brand'][brand_id]
        return brand_name
    elif type(bf_index) == list:
        brand_name = []
        for bf_id in bf_index:
            brand_id = get_brand_from_index(bf_id, str(fam_id), general_database_path, df)
            brands_df = pd.read_csv(path_to_dictionaries+'/brands.csv', sep=";", index_col='brand_id', dtype={'brand_id':int, 'brand':str})
            if brand_id in brands_df.index:
                brand_name.append(brands_df['brand'][brand_id])
            else:
                brand_name.append(None)
        return brand_name
    else:
        return None


# --- Tensorflow getters --- #


def get_segmentation_data(fam_id=None, use_data=None, root_dir=shared_paths.storage_path):
    # Returns absolute paths to shelf image / csv data {path_to_image:path_to_csv}
    #  fam_id: id of family. E.g: '65010000'
    #  use_data: path to client data
    #  root_dir: path to admin data

    shelf_paths = {}
    valid_im_extensions = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')

    # include user shelves if necessary
    if use_data:  # user: client
        csv_files = next(os.walk('{}/Picture_database/im_database_shelf/{}/data'.format(use_data, fam_id)))[2]
        for im_file in next(os.walk('{}/Picture_database/im_database_shelf/{}/images'.format(use_data, fam_id)))[2]:
            csv_file = im_file.split('.', 1)[0]+'.csv'
            if im_file.endswith(valid_im_extensions) and (csv_file in csv_files):
                im_path = '{}/Picture_database/im_database_shelf/{}/images/{}'.format(use_data, fam_id, im_file)
                csv_path = '{}/Picture_database/im_database_shelf/{}/data/{}'.format(use_data, fam_id, csv_file)
                shelf_paths[im_path] = csv_path

    # always include admin shelves
    csv_files = next(os.walk('{}/Picture_database/im_database_shelf/{}/data'.format(root_dir, fam_id)))[2]
    for im_file in next(os.walk('{}/Picture_database/im_database_shelf/{}/images'.format(root_dir, fam_id)))[2]:
        csv_file = im_file.split('.', 1)[0]+'.csv'
        if im_file.endswith(valid_im_extensions) and (csv_file in csv_files):
            im_path = '{}/Picture_database/im_database_shelf/{}/images/{}'.format(root_dir, fam_id, im_file)
            csv_path = '{}/Picture_database/im_database_shelf/{}/data/{}'.format(root_dir, fam_id, csv_file)
            shelf_paths[im_path] = csv_path

    return shelf_paths




#def test():
    # print(root)
    # print("==")
    # print(get_families(root+"/general_database_storage"))
    # print("==")
    # print(get_brands_from_family('10100000', root+"/general_database_storage", df=None))
    # print("==")
    # print(get_brands_from_keyword("essentials", '10100000', root+"/general_database_storage"))
    # print("==")
    # print(get_shelf_images_and_xlsx('65010000', root+"/im_database_shelf"))
    # print("==")
    # print(get_images_from_upc('00688267061370', '10100000', root+"/general_database_storage", root+"/im_database_reference", root+"/im_database", df=None, include_shelf=True, shelf_only=False))
    # print("==")
    # print(get_upc_from_index('3', '98', '10100000', root+"/general_database_storage", df=None))
    # print("==")
    # print(get_index_from_upc('00688267061370', '10100000', root+"/general_database_storage", df=None))
    # print("==")
    # print(get_upc_from_brand('98', '10100000', root+"/general_database_storage", df=None))
    # print("==")
    #print(get_index_from_brand('6500', '65010000', root+"/general_database_storage", df=None))
    # print("==")
    #print(get_brand_name_from_index(['1', '6'], 65010000, root+'/general_database_storage', root+'/dictionaries', df=None))
    #print(get_brand_from_index('6', '65010000', root+"/general_database_storage", df=None))
    #print(get_images_from_upc('05099206058040', '65010000', root+"/general_database_storage", root+"/im_database_reference", root+"/im_database", brand_id='6500', df=None, include_shelf=True, include_ref=True))
    #print(get_price_from_upc("05099206064362", folder="logitech", qopius_storage_path=root, price_range_df=None))
    #print(keyword_in_name("05099206064362", "6500", "65010000", "Bluetooth", im_database_reference=root+'/im_database_reference', df=None))


#test()
