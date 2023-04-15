import pandas as pd
import numpy as np
import haversine as hs
import csv

output_train_filename = 'data/train_with_location.csv'
df_train = pd.read_csv('data/train.csv')
output_test_filename = 'data/test_with_location.csv'
df_test = pd.read_csv('data/test.csv')

commerical_centers = []
with open('data/auxiliary-data/sg-commerical-centres.csv') as f:
  for index, (_, _, lat, long) in enumerate(csv.reader(f)):
    commerical_centers.append((index, lat, long))

df_markets = pd.read_csv('data/auxiliary-data/sg-gov-markets-hawker-centres.csv')
df_population = pd.read_csv('data/auxiliary-data/sg-population-demographics.csv')
df_pri_schools = pd.read_csv('data/auxiliary-data/sg-primary-schools.csv')
df_sec_schools = pd.read_csv('data/auxiliary-data/sg-secondary-schools.csv')
df_malls = pd.read_csv('data/auxiliary-data/sg-shopping-malls.csv')
df_train_stations = pd.read_csv('data/auxiliary-data/sg-train-stations.csv')
MAX = 99999999999999.0

## Features
cbd_dist = 'cbd_dist'
n_center = 'nearest_center' # nearest commerical centres
n_center_dist = 'n_center_dist' # dist to nearest commerical centres

def write_loc_df_to_file():
  df_train.to_csv(output_train_filename, sep='\t')
  df_test.to_csv(output_test_filename, sep='\t')

def dist(p1, p2):
  return hs.haversine(p1, p2)

def dp3(n):
  return "{:.3f}".format(n)

def extract_nearest_commerical_center(lat, long):
  min_dist = MAX
  center = ""
  p1 = (float(lat), float(long))
  for (index, lat, long) in commerical_centers:
    d = dist(p1, (float(lat), float(long)))
    if d < min_dist:
      min_dist = d
      center = index
  return pd.Series([center, dp3(min_dist)])

# we can write a generic function to calculate nearest dist to X
def extract_nearest_location(lat, long, locations_df):
    min_dist = MAX
    nearest_location = ""
    p1 = (float(lat), float(long))
    for index, row in locations_df.iterrows():
        d = dist(p1, (row['lat'], row['lng']))
        if d < min_dist:
            min_dist = d
            nearest_location = index
    return pd.Series([nearest_location, dp3(min_dist)])

def extract_cbd(lat, long):
  p1 = (float(lat), float(long))
  return pd.Series([dist(p1, (1.2867684143873221, 103.85452859811022))])

def main():
    print("==== Extract Location Features ===")

    ## CBD
    print("[1/x] Started processing CBD")
    df_train[[cbd_dist]] = df_train.apply(
        lambda x: extract_cbd(x.latitude, x.longitude), axis=1)
    df_test[[cbd_dist]] = df_test.apply(
        lambda x: extract_cbd(x.latitude, x.longitude), axis=1)
    print("[1/x] Finished processing CBD")

    ## Commerical Centers
    print("[2/x] Started processing sg-commerical-centres")
    df_train[[n_center, n_center_dist]] = df_train.apply(
        lambda x: extract_nearest_commerical_center(x.latitude, x.longitude), axis=1)
    df_test[[n_center, n_center_dist]] = df_test.apply(
        lambda x: extract_nearest_commerical_center(x.latitude, x.longitude), axis=1)
    print("[2/x] Finished processing sg-commerical-centres")

    ## Markets
    print("[3/x] Started processing markets")
    df_train[['nearest_markets', 'nearest_markets_dist']] = df_train.apply(
        lambda x: extract_nearest_location(x.latitude, x.longitude, df_markets), axis=1)
    df_test[['nearest_markets', 'nearest_markets_dist']] = df_test.apply(
        lambda x: extract_nearest_location(x.latitude, x.longitude, df_markets), axis=1)
    print("[3/x] Finished processing markets")    

    ## Schools
    print("[4/x] Started processing df_pri_schools")
    df_train[['nearest_pri_school', 'nearest_pri_school_dist']] = df_train.apply(
        lambda x: extract_nearest_location(x.latitude, x.longitude, df_pri_schools), axis=1)
    df_test[['nearest_pri_school', 'nearest_pri_school_dist']] = df_test.apply(
        lambda x: extract_nearest_location(x.latitude, x.longitude, df_pri_schools), axis=1)
    print("[4/x] Finished processing df_pri_schools")  
    print("[5/x] Started processing df_sec_schools")
    df_train[['nearest_sec_school', 'nearest_sec_school_dist']] = df_train.apply(
        lambda x: extract_nearest_location(x.latitude, x.longitude, df_sec_schools), axis=1)
    df_test[['nearest_sec_school', 'nearest_sec_school_dist']] = df_test.apply(
        lambda x: extract_nearest_location(x.latitude, x.longitude, df_sec_schools), axis=1)
    print("[5/x] Finished processing df_sec_schools")      
    
    ## Malls
    print("[6/x] Started processing df_malls")
    df_train[['nearest_mall', 'nearest_mall_dist']] = df_train.apply(
        lambda x: extract_nearest_location(x.latitude, x.longitude, df_malls), axis=1)
    df_test[['nearest_mall', 'nearest_mall_dist']] = df_test.apply(
        lambda x: extract_nearest_location(x.latitude, x.longitude, df_malls), axis=1)
    print("[6/x] Finished processing df_malls")       

    ## Trains
    print("[7/x] Started processing df_train_stations")
    df_train[['nearest_train_station', 'nearest_train_station_dist']] = df_train.apply(
        lambda x: extract_nearest_location(x.latitude, x.longitude, df_train_stations), axis=1)
    df_test[['nearest_train_station', 'nearest_train_station_dist']] = df_test.apply(
        lambda x: extract_nearest_location(x.latitude, x.longitude, df_train_stations), axis=1)
    print("[7/x] Finished processing df_train_stations")     
    
    ## Population TODO 
    

    write_loc_df_to_file()

if __name__ == "__main__":
    main()
