from env import host, user, password
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
from scipy import stats
from scipy.stats import spearmanr
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score 
from math import sqrt
import os
from pathlib import Path

def get_db_url(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file 
    to create a connection url to access the codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

################################################### Acquire #########################################################
# use this function the 1st time to get initial dataset.
def new_zillow_data():
    '''This function reads in zillow data from Codeup database.'''
    sql_query = '''    
    SELECT properties_2017.fips,
        properties_2017.bedroomcnt,
        properties_2017.bathroomcnt,
        properties_2017.calculatedfinishedsquarefeet,
        properties_2017.taxvaluedollarcnt
        FROM predictions_2017
        JOIN properties_2017
        ON predictions_2017.parcelid = properties_2017.parcelid
        JOIN propertylandusetype
        ON properties_2017.propertylandusetypeid = propertylandusetype.propertylandusetypeid
        WHERE propertylandusetype.propertylandusetypeid = '261'
        AND predictions_2017.transactiondate between '2017-01-01' AND '2017-12-31';'''

    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('zillow'))
    return df

def get_zillow_data():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        #If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv')
        
    else:
        
        #Read fresh data from db into a DataFrame
        df = new_zillow_data()
        
        #Cache data
        df.to_csv('zillow.csv', index=False)
        
    return df

#################################################################################
def prep_zillow_data(df):
    #Replace a whitespace sequence or empty with a NaN value and reassign this manipulation to df.
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    #Drop all rows with any Null values, assign to df, and verify with df.info().
    df = df.dropna()
    
    #rename the columns so they are human readable
    df=df.rename(columns={"bedroomcnt":"bedrooms","bathroomcnt":"bathrooms",
                          "calculatedfinishedsquarefeet":"squarefeet", 
                          "taxvaluedollarcnt": "home_value",})

    # Remove outliers
    df = df[df.bathrooms > 0]
    df = df[df.bathrooms <= 5]
    df = df[df.bedrooms > 0]
    df = df[df.bedrooms <= 5]
    df = df[df.home_value < 1_000_000]
    df = df[df.squarefeet < 5000]
    
    # Convert binary categorical variables to numeric
    df['fips_encoded'] = df.fips.map({6037:1, 6059:2, 6111:3})
    
    df.drop(columns=['fips'], inplace=True)
   
    
   
    
    return df

#################################################################################
def split_zillow_data(df):
    '''
    This function performs split on zillow data.  Purpose of the train, validate, test is to split a dataframe.  
    The train dataset is for training our models. We also perform our exploratory data analysis on train.  
    The validate dataset serves two purposes. First, it is an "out of sample" dataset so that we can evaluate our models 
    on unseen data to measure how well the model generalizes. Second, the validate set allows us to fine tune 
    our hyperparameters.  The test dataset is our final out of sample dataset used to evaluate how well the models 
    tuned on validate generalize on unseen data. Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    return train, validate, test

###################################################################################
def scale_data(train, 
               validate, 
               test, 
               columns_to_scale = ['bedrooms', 'bathrooms', 'squarefeet'],
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                           columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
    
################################################## viz scaler############################################################
def visualize_scaler(scaler, df, columns_to_scale, bins=10):
    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(16,9))
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()
#return fig, axs
##########################

def wrangle_zillow():
    df=get_zillow_data()
    df=prep_zillow_data(df)
    train, validate, test=split_zillow_data(df)
    scaler, train_scaled, validate_scaled, test_scaled = scale_data(train, validate, test, return_scaler=True)

    return train, validate, test

##################################################################