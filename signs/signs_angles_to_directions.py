import ast
import json
import math
from pathlib import Path

import geopy.distance
import numpy as np
import pandas as pd
from geojson import Feature, Point
from sklearn.neighbors import KDTree

def angle_diff_degrees(angle1, angle2):
    """
    Calculates the shortest angular difference between two angles in degrees.
    The result will be in the range of -180 to 180 degrees.
    """
    diff = angle2 - angle1
    diff = (diff + 180) % 360 - 180
    return abs(diff)

def get_rotation_angle(base_point, comp_point):
    dx = base_point[0] - comp_point[0]
    dy = base_point[1] - comp_point[1]
    theta = math.atan2(dx, dy)
    ang = theta*180/math.pi
    while ang < 0:
        ang+=360
    return ang

def point(lon: float, lat: float, hdg: float, properties: dict, length: float = 5) -> Feature:
    return Feature(geometry=Point((float(lon), float(lat))), properties=properties)

duplicates = []
csv = pd.read_csv("/Users/hlebtkach/Downloads/nodes.csv")
intersection_nodes = {}
for index, val in csv.iterrows():
    d = val['feature']
    res = json.loads(d)
    tmp = []
    info = {}
    st = set()
    ls = ["motorway", "trunk", "motorway_link", "trunk_link", "primary", "primary_link", "secondary",
              "secondary_link", "tertiary", "unclassified", "residential", "tertiary_link", "service"]
    st.update(ls)

count = 0

string = '{"type": "FeatureCollection", "features": ['
delete = set()
for index, val in csv.iterrows():
    d = val['feature']
    res = json.loads(d)
    tmp = []
    info = {}
    prev_direction = ''
    after_intersection = False
    if (res['properties'] is None) or ('highway' not in res['properties']) or (res['properties']['highway'] not in st):
        continue
    for node_idx in range(0, len(res['nodes'])):
        current_node = res['nodes'][node_idx]
        if ('direction' in current_node['properties'] and 'highway' in current_node['properties']
                and (current_node['properties']['highway'] == 'stop'
                or current_node['properties']['highway'] == 'give_way'
                or current_node['properties']['highway'] == 'traffic_signals')):
            next_angle = 0
            prev_angle = 0

            if node_idx != len(res['nodes']) - 1:
                next_angle = get_rotation_angle(
                    res['nodes'][node_idx + 1]['geometry']['coordinates'],
                    current_node['geometry']['coordinates'])
            else:
                next_angle = get_rotation_angle(
                    current_node['geometry']['coordinates'],
                    res['nodes'][node_idx - 1]['geometry']['coordinates'])

            if node_idx != 0:
                prev_angle = get_rotation_angle(
                    res['nodes'][node_idx - 1]['geometry']['coordinates'],
                    current_node['geometry']['coordinates'])
            else:
                prev_angle = get_rotation_angle(
                    current_node['geometry']['coordinates'],
                    res['nodes'][node_idx + 1]['geometry']['coordinates'])

            dir = 0
            try:
                dir = int(current_node['properties']['direction'])
            except:                   # not angle
                continue
            delta_theta1 = angle_diff_degrees(dir, next_angle)
            delta_theta2 = angle_diff_degrees(dir, prev_angle)

            if delta_theta1 > delta_theta2:
                current_node['properties']['alt_dir'] = 'forward'
            else:
                current_node['properties']['alt_dir'] = 'backward'

            current_node['properties']['id'] = current_node['id']
            feature = (
                point(
                    current_node['geometry']['coordinates'][0],
                    current_node['geometry']['coordinates'][1],
                    0,
                    properties=current_node['properties'],
                )
            )
            string += json.dumps(feature)
            string += ','

coords = []
for node in duplicates:
    coords.append(node['geometry']['coordinates'])

string = string[:-1]
string+="]}"
with open("/Users/hlebtkach/Downloads/all_stops.geojson", 'w', encoding='utf-8') as w:
    w.write(string)