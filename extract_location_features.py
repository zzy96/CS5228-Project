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
# df_population = pd.read_csv('data/auxiliary-data/sg-population-demographics.csv')
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

def preprocess_demographic_data():
    age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44','45-49','50-54','55-59', '60-64', '65-69', '70-74',  '75-79', '80-84', '85+']
    n = len(age_groups) + 1 # last col for total pop
    data = {}
    with open('data/auxiliary-data/sg-population-demographics.csv') as f:
        for index, (area, subzone, group, _, count) in enumerate(csv.reader(f)):
            if index == 0:
                continue
            # ignore M / F split (not likely to be important
            if area not in data:
                data[area] = {}
            if subzone not in data[area]:
                data[area][subzone] = [0] * n

            ind = age_groups.index(group)
            data[area][subzone][ind] += int(count)
            data[area][subzone][-1] += int(count)
    return data

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
    
    ## Population
    print("[8/x] Started processing sg-population-demographics")
    # demographic_data['yishun']['nee soon'] => [140, 180, 220, 260, 360, 290, 190, 240, 250, 250, 310, 330, 300, 250, 140, 80, 50, 50]
    demographic_data = preprocess_demographic_data() 
    n = 19
    pop_keys = ['pop' + str(x) for x in range(n)]
    # cityhall got no pop data, to sub with central subzone
    missing =  demographic_data['downtown core']['central subzone']
    df_train[pop_keys] = df_train.apply(
        lambda x: pd.Series(demographic_data[x.planning_area][x.subzone]) if (x.planning_area in demographic_data) and (x.subzone in demographic_data[x.planning_area]) else missing, axis=1)
    df_test[pop_keys] = df_test.apply(
        lambda x: pd.Series(demographic_data[x.planning_area][x.subzone]) if (x.planning_area in demographic_data) and (x.subzone in demographic_data[x.planning_area]) else missing, axis=1)
    print("[8/x] Finished processing sg-population-demographics")     

    write_loc_df_to_file()

if __name__ == "__main__":
    main()