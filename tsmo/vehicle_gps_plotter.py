import gmplot
from csv import reader
import pandas as pd
import numpy as np
import math
from statistics import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.ticker import PercentFormatter
import math
import geopy.distance
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
import constants

def distanceBetween(txLat, txLong, rxLat, rxLong):
    txCoordinates = (float(txLat), float(txLong))
    rxCoordinates = (float(rxLat), float(rxLong))
    return geopy.distance.distance(txCoordinates, rxCoordinates).m

def distanceHelper(row, stopLat, stopLon):
    return distanceBetween(stopLat, stopLon, row['field.core_data.latitude'], row['field.core_data.longitude'])

data = pd.read_csv(f'../data/{constants.OUTGOING_BSM_DIR}/bsm.csv')
#print(data['field.core_data.speed'])
#west left: (38.9550182, -77.1494752),
#west right: (38.9549927, -77.1494879)
#east left: (38.9549458, -77.1491373)
#east right: (38.9549741, -77.1491307)

stopBarLat = 38.9550182
stopBarLon = -77.1494752

lat = data['field.core_data.latitude']
lon = data['field.core_data.longitude']
sec = data['field.core_data.sec_mark']

data['distance_to_stop_bar(m)'] = data.apply(lambda row: distanceHelper(row, stopBarLat, stopBarLon), axis=1)
data['vehicle_speed_converted(m/s)'] = data['field.core_data.speed']*0.02

test = data[(data['distance_to_stop_bar(m)'] < 15)]
#test = data[(data['distance_to_stop_bar(m)'] < 15)&(data['field.core_data.speed'] > 0)&(data['field.core_data.speed'] < 5)]
test.to_csv(f'test.csv', index=False)
#print(test)

lat = test['field.core_data.latitude']
lon = test['field.core_data.longitude']
sec = test['field.core_data.sec_mark']

color=['red', 'blue', 'green', 'green', 'blue', 'purple', 'black']
key = 'AIzaSyByQ-siatijl7bGnkLB4FaYx64zEHc236I'

vehicle_map = gmplot.GoogleMapPlotter(stopBarLat, stopBarLon, 18, apikey=key)
vehicle_map.scatter([stopBarLat], [stopBarLon], color[0], size = 5, marker = True)
vehicle_map.scatter(lat, lon, color[2], size = 5, marker = True)
vehicle_map.draw(f'../data/{constants.PLOT_DIR}/West_Left.html')

print(test['distance_to_stop_bar(m)'].min())
