import get_data as GD
import sys,os
import idw as idw
import fwi as fwi
import json

dirname = '' #Insert data directory path or #dirname = os.getcwd()
file_path_hourlyf = os.path.join(dirname, 'datasets/weather/hourly_feather/')
file_path_hourly = os.path.join(dirname, 'datasets/weather/hourly_csv/')
file_path_daily = os.path.join(dirname, 'datasets/weather/all_daily/')
shapefile = os.path.join(dirname, 'datasets/study_area/QC_ON_albers_dissolve.shp')

file_path_elev = os.path.join(dirname,'datasets/lookup_files/elev_csv_200km.csv')
idx_list = GD.get_col_num_list(file_path_elev,'elev')

dictionary = fwi.calc_duration_in_ecozone(file_path_daily,file_path_hourlyf,file_path_elev,\
                         idx_list,shapefile,\
                         ['boreal1_ecozone61','boreal2_easternC5','hudson','taiga_shield'],1922,2019,'GPR',True,False)
print(dictionary)
year=[]
eco1=[]
eco2=[]
eco3=[]
eco4=[]
sub_key = ['boreal1_ecozone61','boreal2_easternC5','hudson','taiga_shield']
for key in dictionary:
    eco1.append(dictionary[key][sub_key[0]])
    eco2.append(dictionary[key][sub_key[1]])
    eco3.append(dictionary[key][sub_key[2]])
    eco4.append(dictionary[key][sub_key[3]])
    year.append(key)

rows = zip(year,eco1,eco2,eco3,eco4)
import csv
export_path = '' #insert path + output file name to save the information to 
with open(export_path, "w") as f:
    writer = csv.writer(f,lineterminator = '\n')
    for row in rows:
        writer.writerow(row)
