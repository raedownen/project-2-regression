from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from pathlib import Path
from env import host, user, password


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

def wrangle_zillow():
    df=get_zillow_data()
    df=prep_zillow_data(df)
    train, validate, test=split_zillow_data(df)

    return train, validate, test