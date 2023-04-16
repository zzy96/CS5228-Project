import pandas as pd
import numpy as np

housing_indices_dict = {}

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

def process_year_month(raw):
    [year, month] = raw.split('-')
    month = int(month)
    if 1 <= month and month <= 3:
        key = year + '-Q1'
    elif 4 <= month and month <= 6:
        key = year + '-Q2'
    elif 7 <= month and month <= 9:
        key = year + '-Q3'
    else:
        key = year + '-Q4'
    return pd.Series((housing_indices_dict[key], year, month))

def preprocess(df, df_pop = None):
    # process year month and housing price index based on year and month
    df[['price_index', 'year', 'month']] = df['month'].apply(process_year_month)

    # flat_type
    df[['num_rooms', 'is_executive', 'is_multi_gen']] = df['flat_type'].apply(parse_flat_type)

    # storey_range
    # create a new binary variable for low floor (1-6) which accounts for 40%+ of the data
    df['storey_range_avg'] = df['storey_range'].str.split(
        ' to ').apply(lambda x: (int(x[1])+int(x[0]))/2)
    # this range may be accessible by stairs so may be special
    df['is_low_floor'] = df['storey_range_avg'].apply(lambda x: 1 if x < 6 else 0)

    # lease_commence_date
    # df['lease_commence_date'] = df['lease_commence_date'].apply(lambda x: 2023 - x)

    # convert string to categorical variables
    df['town'] = df['town'].astype('category')
    df['block'] = df['block'].astype('category')
    df['street_name'] = df['street_name'].astype('category')
    df['flat_model'] = df['flat_model'].astype('category')
    df['subzone'] = df['subzone'].astype('category')
    df['planning_area'] = df['planning_area'].astype('category')
    df['region'] = df['region'].astype('category')

    if df_pop is not None:
        pop_columns = ['pop0', 'pop1', 'pop2', 'pop3', 'pop4', 'pop5', 'pop6', 'pop7', 'pop8',
                       'pop9', 'pop10', 'pop11', 'pop12', 'pop13', 'pop14', 'pop15', 'pop16', 'pop17', 'pop18']
        df = df.drop(columns=pop_columns)
        return pd.concat([df, df_pop[pop_columns]], axis=1)

    return df

def main():
    df = pd.read_csv(
        'data/gov/housing-and-development-board-resale-price-index-1q2009-100-quarterly.csv')
    for _, row in df.iterrows():
        # Use the first column as the key and the second column as the value
        housing_indices_dict[row['quarter']] = row['index']
    final_columns_train = [
        'price_index', 'year', 'month', 'num_rooms', 'is_executive', 'is_multi_gen',
        'storey_range_avg', 'is_low_floor', 'floor_area_sqm', 'lease_commence_date', 'latitude', 'longitude', 'elevation',
        'town', 'block', 'street_name', 'flat_model', 'subzone', 'planning_area', 'region',
        'cbd_dist', 'nearest_center', 'n_center_dist',
        'nearest_markets', 'nearest_markets_dist', 'nearest_pri_school', 'nearest_pri_school_dist', 'nearest_sec_school', 'nearest_sec_school_dist', 'nearest_mall', 'nearest_mall_dist', 'nearest_train_station', 'nearest_train_station_dist',
        'pop0', 'pop1', 'pop2', 'pop3', 'pop4', 'pop5', 'pop6', 'pop7', 'pop8', 'pop9', 'pop10', 'pop11', 'pop12', 'pop13', 'pop14', 'pop15', 'pop16', 'pop17', 'pop18',
        'resale_price'
    ]
    final_columns_test = [
        'price_index', 'year', 'month', 'num_rooms', 'is_executive', 'is_multi_gen',
        'storey_range_avg', 'is_low_floor', 'floor_area_sqm', 'lease_commence_date', 'latitude', 'longitude', 'elevation',
        'town', 'block', 'street_name', 'flat_model', 'subzone', 'planning_area', 'region',
        'cbd_dist', 'nearest_center', 'n_center_dist',
        'nearest_markets', 'nearest_markets_dist', 'nearest_pri_school', 'nearest_pri_school_dist', 'nearest_sec_school', 'nearest_sec_school_dist', 'nearest_mall', 'nearest_mall_dist', 'nearest_train_station', 'nearest_train_station_dist',
        'pop0', 'pop1', 'pop2', 'pop3', 'pop4', 'pop5', 'pop6', 'pop7', 'pop8', 'pop9', 'pop10', 'pop11', 'pop12', 'pop13', 'pop14', 'pop15', 'pop16', 'pop17', 'pop18'
    ]
    df = pd.read_csv('data/train_with_location.csv', sep='\t')
    df_pop = pd.read_csv('data/train_with_pop.csv', sep='\t')
    preprocess(df, df_pop)[final_columns_train].to_csv(
        'data/train_preprocessed.csv', index=False, sep='\t')
    # df = pd.read_csv('data/test_with_location.csv', sep='\t')
    # preprocess(df)[final_columns_test].to_csv(
    #     'data/test_preprocessed.csv', index=False, sep='\t')

if __name__ == "__main__":
    main()
