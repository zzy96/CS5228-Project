import pandas as pd
import numpy as np

def parse_flat_type(s_type):
    if not isinstance(s_type, str):
        return pd.Series((None, None, None))
    if s_type[0] in ['1', '2', '3', '4', '5']:
        return pd.Series((int(s_type[0]), False, False))
    elif s_type == 'executive':
        return pd.Series((0, True, False))
    elif s_type == 'multi generation':
        return pd.Series((0, False, True))
    else:
        return pd.Series((None, None, None))
    

def preprocess(df):
    # process year month
    df['year'] = df['month'].str.split('-').apply(lambda x:x[0]).astype('int')
    df['month'] = df['month'].str.split('-').apply(lambda x:x[1]).astype('int')

    # flat_type
    df[['num_rooms', 'is_executive', 'is_multi_gen']] = df['flat_type'].apply(parse_flat_type)

    # storey_range
    # create a new binary variable for low floor (1-6) which accounts for 40%+ of the data
    df['storey_range_avg'] = df['storey_range'].str.split(
        ' to ').apply(lambda x: (int(x[1])+int(x[0]))/2)
    # this range may be accessible by stairs so may be special
    df['is_low_floor'] = df['storey_range_avg'].apply(lambda x: 1 if x < 6 else 0)

    # convert string to categorical variables
    df['town'] = df['town'].astype('category')
    df['block'] = df['block'].astype('category')
    df['street_name'] = df['street_name'].astype('category')
    df['flat_model'] = df['flat_model'].astype('category')
    df['subzone'] = df['subzone'].astype('category')
    df['planning_area'] = df['planning_area'].astype('category')
    df['region'] = df['region'].astype('category')

    return df


final_columns_train = [
    'year', 'month', 'num_rooms', 'is_executive', 'is_multi_gen',
    'storey_range_avg', 'is_low_floor', 'floor_area_sqm', 'lease_commence_date', 'latitude', 'longitude', 'elevation',
    'town', 'block', 'street_name', 'flat_model', 'subzone', 'planning_area', 'region',
    'resale_price'
]
final_columns_test = [
    'year', 'month', 'num_rooms', 'is_executive', 'is_multi_gen',
    'storey_range_avg', 'is_low_floor', 'floor_area_sqm', 'lease_commence_date', 'latitude', 'longitude', 'elevation',
    'town', 'block', 'street_name', 'flat_model', 'subzone', 'planning_area', 'region'
]

df = pd.read_csv('data/train.csv')
preprocess(df)[final_columns_train].to_csv(
    'data/train_preprocessed.csv', index=False)
df = pd.read_csv('data/test.csv')
preprocess(df)[final_columns_test].to_csv(
    'data/test_preprocessed.csv', index=False)
