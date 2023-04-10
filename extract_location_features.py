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

    ## Population

    ## Schools

    ## Malls

    ## Trains

    write_loc_df_to_file()

if __name__ == "__main__":
    main()
