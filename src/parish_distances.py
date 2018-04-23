import time
from math import radians, cos, sin, asin, sqrt
import os
import cPickle as pickle
import numpy as np

import hiski_models as hiski
from etrs2lonlat import etrs2lonlat


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    
    Source: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def get_distance_map():
    t0 = time.time()
    D = {}
    parishes = hiski.Parish.query.all()
    for p1i, p1 in enumerate(parishes):
        id1 = p1.id
        D[(id1, id1)] = 0
        if p1.lon is not None and p1.lat is not None:
            lon1, lat1 = etrs2lonlat(p1.lon, p1.lat)
        for p2 in parishes[p1i+1:]:
            if p1.lon is not None and p1.lat is not None and \
                    p2.lon is not None and p2.lat is not None:
                lon2, lat2 = etrs2lonlat(p2.lon, p2.lat)
                dist = haversine(lon1, lat1, lon2, lat2)
            else:
                dist = None
            id2 = p2.id
            D[(id1, id2)] = dist
            D[(id2, id1)] = dist
    print "Computing all distances took {:.2f} seconds.".format(time.time() - t0)
    return D
            

def load_or_compute_distance_map(path, do_save=True):
    if os.path.isfile(path):
        try:
            distance_map = pickle.load(open(path, 'rb'))
        except:
            distance_map = get_distance_map()
    else:
        distance_map = get_distance_map()
        if do_save:
            pickle.dump(distance_map, open(path, 'wb'))
    return distance_map


def get_parish_coordinates():
    t0 = time.time()
    D = {}
    parishes = hiski.Parish.query.all()
    avg_lon = np.mean([p.lon for p in parishes if p.lon is not None])
    avg_lat = np.mean([p.lat for p in parishes if p.lat is not None])
    for p in parishes:
        lon = p.lon
        lat = p.lat
        if lon is None:
            lon = avg_lon
        if lat is None:
            lat = avg_lat
        D[p.id] = (lon, lat)
    D['avg_coord'] = (avg_lon, avg_lat)
    #print "Retrieving all coordinates took {:.2f} seconds.".format(time.time() - t0)
    return D
