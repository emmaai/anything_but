import pandas as pd
import numpy as np
from wit_tooling import construct_product, query_datasets
from pyproj import Proj, transform
from datacube import Datacube
from datacube.virtual.transformations import MakeMask, ApplyMask

def fc_location(dc, fc_product, site_index, lat, lon):
    inproj = Proj("epsg:4326")
    outproj = Proj("epsg:3577")
    x, y = transform(inproj, outproj, lat, lon)
    x -= x%25
    y -= y%25
    query = {'time':('1987-01-01', '2020-12-31'),
            'x':(x, x+25), 'y':(y, y+25), 'crs':'EPSG:3577'}
    datasets = fc_product.query(dc, **query)
    grouped = fc_product.group(datasets, **query)
    results = load_data(fc_product, grouped)
    results = results.where(results > -1)
    results.to_dataframe().dropna().to_csv(str(int(site_index))+'.csv')

def load_data(fc_product, grouped, mask_by_wofs=True):
    results = fc_product.fetch(grouped)
    results = ApplyMask('pixelquality', apply_to=['BS', 'PV', 'NPV']).compute(results)
    water_mask = results.water.to_dataset()
    results = results.drop('water')
    if mask_by_wofs:
        flags = {'cloud': False,
                'cloud_shadow': False,
                'noncontiguous': False,
                'water_observed': False
                }
        water_mask = MakeMask('water', flags).compute(water_mask)
        results = results.merge(water_mask)
        results = ApplyMask('water', apply_to=['BS', 'PV', 'NPV']).compute(results)
    return results

def main():
    location_list = pd.read_csv("Locations_of_cp_sites.csv")
    fc_product = construct_product('fc_pd.yaml')
    dc = Datacube()
    for index, location in location_list.iterrows():
        if np.isnan(location['Latitude']) or np.isnan(location['Longitude']):
            continue
        else:
            fc_location(dc, fc_product, location['ALL SITES'], location['Latitude'], location['Longitude'])
            print("compute", location['ALL SITES'])

if __name__ == "__main__":
    main()
