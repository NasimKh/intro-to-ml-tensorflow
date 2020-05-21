import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


#import funcs as fn
# magic word for producing visualizations in notebook
#%matplotlib inline



def replace_with_nans(feat_info,azdias) :
    for cols in azdias.columns :

        unknown_val = feat_info.loc[feat_info.attribute== cols ,'missing_or_unknown'].values[0]
        vals = unknown_val.split('[', 1)[1].split(']')[0].split(',')
        missing_vals = [int(x) if x.lstrip('-').isdigit() else x for x in vals]

        for val in missing_vals :
            azdias[cols].replace(to_replace= val , value= np.NaN ,inplace=True)

    return azdias


def outlier_cols(azdias):
    threshhold = 200000
    df=pd.DataFrame(azdias.isnull().sum(), columns = ['nulls']  ).reset_index()
    outlier_cols=df[df.nulls>threshhold]['index'].values
    return outlier_cols


def outlier_rows(azdias,drop_percentage ):
    cols_num= azdias.shape[1]
    threshhold = int(cols_num * drop_percentage/100)
    #df=pd.DataFrame(azdias.isnull().sum(axis=1))
    #df.sort_values(by = 0 , ascending = False)
    drop_rows= azdias[azdias.isnull().sum(axis= 1)>threshhold].index
    return drop_rows


def drop_col_row(outlier_cols,drop_rows,df):
    categorical_drop_cols = ['CAMEO_DEU_2015' , 'LP_FAMILIE_FEIN' , 'LP_STATUS_FEIN']
    dropped_cols = categorical_drop_cols+list(outlier_cols)

    df = df.drop(columns=dropped_cols )
    df = df.drop(axis= 0 , index = drop_rows)
    return df , dropped_cols


def to_bool(df):
   #df=  df.OST_WEST_KZ.replace({'W': 0, 'O': 1} )
    bool_types = ['OST_WEST_KZ','VERS_TYP','SOHO_KZ','GREEN_AVANTGARDE','ANREDE_KZ']
    for bol in bool_types:
        df[bol]=  df[bol].replace({df[bol].dropna().unique()[0]: 0, df[bol].dropna().unique()[1]: 1} )
    return df , bool_types

def choose_encode_col(dropped_cols , feat_info ,bool_types):
    encoding_cols=feat_info[(feat_info.type == 'categorical') &
                                (~feat_info.attribute.isin( bool_types))&
                                (~feat_info.attribute.isin( dropped_cols))].attribute.values
    return encoding_cols

def one_hot_df(encoding_cols ,df ):
    one_hot_df =pd.get_dummies(data=df, columns=encoding_cols
               ,drop_first=True )
    df = pd.concat([df, one_hot_df], axis=1, sort=False)
    df_onehot_encoded = df.loc[:,~df.columns.duplicated()]
    return df_onehot_encoded

def feature_engineer (df_onehot_encoded):
    Mainstream = [1 , 3 ,5 , 8 , 10 , 12 , 14]

    df_onehot_encoded ['mainstream']= df_onehot_encoded['PRAEGENDE_JUGENDJAHRE'].isin(Mainstream).astype(int)
    decade_dic = { 1: 40 , 2: 40, 3: 50, 4: 50, 5: 60,
                   6: 60, 7: 60, 8: 70, 9: 70,10: 80,
                   11: 80,12: 80,13: 80,14: 90,15: 90}
    df_onehot_encoded['decade'] = df_onehot_encoded['PRAEGENDE_JUGENDJAHRE'].map(decade_dic)
    df_onehot_encoded['CAMEO_INTL_2015']=df_onehot_encoded['CAMEO_INTL_2015'].astype(float)
    df_onehot_encoded['family_status']=df_onehot_encoded['CAMEO_INTL_2015']%10
    df_onehot_encoded['wealth_class']=df_onehot_encoded['CAMEO_INTL_2015']//10
    df_onehot_encoded.drop(columns = ['CAMEO_INTL_2015','PRAEGENDE_JUGENDJAHRE' ], inplace = True)
    return df_onehot_encoded


def impute_missing_vals(df_onehot_encoded):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_result=imp.fit_transform(result)
    return imputed_result


def scale_vals(imputed_result):
    scaler = StandardScaler()
    scaled_result=scaler.fit_transform(imputed_result)
    return scaled_result











