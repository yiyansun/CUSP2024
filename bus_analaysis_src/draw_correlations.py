from gmplot import GoogleMapPlotter
import csv
import pandas as pd
import json

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from gmplot import GoogleMapPlotter
from random import random


class CustomGoogleMapPlotter(GoogleMapPlotter):
    def __init__(self, center_lat, center_lng, zoom, apikey='',
                 map_type='satellite'):
        if apikey == '':
            try:
                with open('apikey.txt', 'r') as apifile:
                    apikey = apifile.readline()
            except FileNotFoundError:
                pass
        super().__init__(center_lat, center_lng, zoom, apikey)

        self.map_type = map_type
        assert(self.map_type in ['roadmap', 'satellite', 'hybrid', 'terrain'])

    def write_map(self,  f):
        f.write('\t\tvar centerlatlng = new google.maps.LatLng(%f, %f);\n' %
                (self.center[0], self.center[1]))
        f.write('\t\tvar myOptions = {\n')
        f.write('\t\t\tzoom: %d,\n' % (self.zoom))
        f.write('\t\t\tcenter: centerlatlng,\n')

        # Change this line to allow different map types
        f.write('\t\t\tmapTypeId: \'{}\'\n'.format(self.map_type))

        f.write('\t\t};\n')
        f.write(
            '\t\tvar map = new google.maps.Map(document.getElementById("map_canvas"), myOptions);\n')
        f.write('\n')

    def color_scatter(self, lats, lngs, values=None, colormap='coolwarm',
                      size=None, marker=False, s=None, **kwargs):
        def rgb2hex(rgb):
            """ Convert RGBA or RGB to #RRGGBB """
            rgb = list(rgb[0:3]) # remove alpha if present
            rgb = [int(c * 255) for c in rgb]
            hexcolor = '#%02x%02x%02x' % tuple(rgb)
            return hexcolor

        if values is None:
            colors = [None for _ in lats]
        else:
            cmap = plt.get_cmap(colormap)
            norm = Normalize(vmin=min(values), vmax=max(values))
            scalar_map = ScalarMappable(norm=norm, cmap=cmap)
            colors = [rgb2hex(scalar_map.to_rgba(value)) for value in values]
        for lat, lon, c in zip(lats, lngs, colors):
            self.scatter(lats=[lat], lngs=[lon], c=c, size=size, marker=marker,
                         s=s, **kwargs)


def draw():
    sensors = json.loads(open("C:/Users/Sergey/Desktop/CUSP2024/data/sensors", 'r').read())
    coords = {}
    for site in sensors["Sites"]["Site"]:
        coords[site["@SiteCode"]] = (site["@Latitude"], site["@Longitude"])

    df = pd.read_csv("C:/Users/Sergey/Desktop/CUSP2024/data/correlation_pearson.csv")

    lats = []
    longs = []
    vals = []
    for site in df["sitecode"]:
        lats.append(float(coords[site][0]))
        longs.append(float(coords[site][1]))
    for val in df["speed_no_lag_corr"]:
        vals.append(float(val))

    gmap = CustomGoogleMapPlotter(51.5075819137762, -0.133068372704093, 5, map_type='satellite')
    gmap.color_scatter(lats, longs, vals, colormap='coolwarm')
    gmap.draw("map.html")
draw()