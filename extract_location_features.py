import pandas as pd
import numpy as np
import haversine as hs
import csv

output_filename = 'data/train_with_location.csv'
df = pd.read_csv('data/train.csv')
commerical_centers = []
with open('data/auxiliary-data/sg-commerical-centres.csv') as f:
  for index, (_, _, lat, long) in enumerate(csv.reader(f)):
    commerical_centers.append((index, lat, long))

df_markets = pd.read_csv('data/auxiliary-data/sg-gov-markets-hawker-centres.csv')
df_population = pd.read_csv('data/auxiliary-data/sg-population-demographics.csv')
df_pri_schools = pd.read_csv('data/auxiliary-data/sg-primary-schools.csv')
df_sec_schools = pd.read_csv('data/auxiliary-data/sg-secondary-schools.csv')
df_malls = pd.read_csv('data/auxiliary-data/sg-shopping-malls.csv')
df_train = pd.read_csv('data/auxiliary-data/sg-train-stations.csv')
MAX = 99999999999999.0

## Features
n_center = 'nearest_center' # nearest commerical centres
n_center_dist = 'n_center_dist' # dist to nearest commerical centres

def write_train_with_loc_df_to_file():
  df.to_csv(output_filename, sep='\t')

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

def main():
    print("==== Extract Location Features ===")

    ## Commerical Centers
    print("[1/x] Started processing sg-commerical-centres")
    df[[n_center, n_center_dist]] = df.apply(lambda x: extract_nearest_commerical_center(x.latitude, x.longitude), axis=1)
    print("[1/x] Finished processing sg-commerical-centres")

    ## Markets

    ## Population

    ## Schools

    ## Malls

    ## Trains

    write_train_with_loc_df_to_file()

if __name__ == "__main__":
    main()