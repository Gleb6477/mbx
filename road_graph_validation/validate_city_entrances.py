import json

import geopy.distance
import numpy as np
import pandas as pd
from geojson import Feature, FeatureCollection
from scipy.spatial import KDTree

from shapely import Point, LineString

ways = {}
results = set()
nodes = {}
node_refs = {}

def get_distance_in_meters(from_point: np.array, to_point: np.array) -> int:
    from_point, to_point = reversed(from_point), reversed(to_point)
    return geopy.distance.distance(from_point, to_point).m

def parse_features(feature):
    way_tags = feature['properties']
    way_coords = feature['geometry']['coordinates']
    way_id = feature['id']
    nds = []

    for node in feature['nodes']:
        node_tags = node['properties']
        node_coords = node['geometry']['coordinates']
        node_id = node['id']
        node_obj = Node(node_id, node_tags, node_coords)
        nodes[node_id] = node_obj
        if node_id in node_refs:
            node_refs[node_id].append(feature['id'])
        else:
            node_refs[node_id] = [feature['id']]
        nds.append(node_obj)
    ways[way_id] = Way(way_id, way_tags, nds, way_coords)

def get_distance_to_closest_split_nodes(way, node_idx):
    start_dist = get_distance(way.get_nodes()[0].geometry, way.get_nodes()[node_idx].geometry)
    end_dist = get_distance(way.get_nodes()[len(way.get_nodes())-1].geometry, way.get_nodes()[node_idx].geometry)
    return  way.get_nodes()[0], start_dist, way.get_nodes()[len(way.get_nodes())-1], end_dist

def get_distance(start_node, end_node):
    return geopy.distance.distance(start_node, end_node).m

class Way:

    def __init__(self, id, tags, nodes, geometry):
        self.id = id
        self.tags = tags
        self.nodes = nodes
        self.geometry = geometry

    def get_nodes(self):
        return self.nodes

    def is_node_first(self, id):
        return self.nodes[0].id == id

    def is_node_last(self, id):
        return self.nodes[len(self.nodes)-1].id == id

class Node:

    def __init__(self, id, geometry=None, tags=None):
        self.id = id
        self.tags = tags
        self.geometry = geometry

def filter_close_points_kdtree(points, max_distance_m):
    if not points:
        return []
    points_array = np.array(points)
    points_rad = np.radians(points_array)
    points_scaled = points_rad * 6371
    tree = KDTree(points_scaled)
    pairs = tree.query_pairs(max_distance_m, output_type='ndarray')
    to_remove = set()
    for i, j in pairs:
        actual_dist = get_distance(
            points_array[i],points_array[j]
        )
        if actual_dist <= max_distance_m:
            to_remove.add(j)
    filtered_points = [points[i] for i in range(len(points)) if i not in to_remove]
    return filtered_points

def dfs(initial_node, prev_local_node, local_node, distance_left, level, visited):
    if prev_local_node is not None and prev_local_node.id == local_node.id:
        return
    global results
    adjacent_ways = node_refs[local_node.id]
    for way_id in adjacent_ways:
        if way_id in visited:
            continue
        way = ways[way_id]
        start = way.get_nodes()[0]
        end = way.get_nodes()[len(way.get_nodes())-1]
        ln = get_distance(start.geometry, end.geometry)
        visited.add(way_id)
        if ln <= distance_left:
            distance_left-=ln
            if 'maxspeed' in way.tags:
                prev_local_node = local_node
                if start.id != local_node.id:
                    local_node = start
                else:
                    local_node = end
                try:
                    if int(way.tags['maxspeed']) <= 50:
                        results.remove(initial_node.id)
                        return
                    else:
                        dfs(initial_node, prev_local_node, local_node, distance_left, level+1, visited)
                except:
                    try:
                        if prev_local_node.id == way.nodes[0].id:
                            if int(way.tags['maxspeed:forward']) <= 50:
                                results.remove(initial_node.id)
                                return
                        elif prev_local_node.id == way.nodes[len(way.nodes)-1].id:
                            if int(way.tags['maxspeed:backward']) <= 50:
                                results.remove(initial_node.id)
                                return
                    except:
                        None
            dfs(initial_node, prev_local_node, local_node, distance_left, level+1, visited)
        else:
            if 'maxspeed' in way.tags:
                try:
                    if int(way.tags['maxspeed']) <= 50:
                        results.remove(initial_node.id)
                        return
                except:
                    try:
                        if prev_local_node.id == way.nodes[0].id:
                            if int(way.tags['maxspeed:forward']) <= 50:
                                results.remove(initial_node.id)
                                return
                        elif prev_local_node.id == way.nodes[len(way.nodes)-1].id:
                            if int(way.tags['maxspeed:backward']) <= 50:
                                results.remove(initial_node.id)
                                return
                    except:
                        None
            [tmp.append(ways[x]) for x in visited]

if __name__ == "__main__":
    roads = pd.read_csv("/Users/hlebtkach/Downloads/france_ways.csv")
    output_ways = []
    tmp = []
    visited_nodes = set()
    for idx, row in roads.iterrows():
        parse_features(json.loads(row.feature))
    for way in ways.items():
        if 'junction' in way[1].tags and (way[1].tags['junction'] == 'roundabout' or way[1].tags['junction'] == 'circular'):
            continue
        for node_idx in range(0, len(way[1].get_nodes())):
            node = way[1].get_nodes()[node_idx]
            tags = node.tags
            initial_distance = 50.0
            if 'traffic_sign' in tags and tags['traffic_sign'] == 'city_limit':
                if node.id in visited_nodes:
                    continue
                results.add(node.id)
                if not (way[1].is_node_first(node.id) and way[1].is_node_last(node.id)):
                    if 'maxspeed' in way[1].tags:
                        try:
                            if int(way[1].tags['maxspeed']) <= 50:
                                results.remove(node.id)
                                continue
                        except:
                            None
                    elif 'maxspeed' not in way[1].tags or way[1].tags['maxspeed'] == '[DELETED]':
                        if 'maxspeed:forward' in way[1].tags:
                            try:
                                if int(way[1].tags['maxspeed:forward']) <= 50:
                                    results.remove(node.id)
                                    continue
                            except:
                                None
                        if 'maxspeed:backward' in way[1].tags:
                            try:
                                if int(way[1].tags['maxspeed:backward']) <= 50:
                                    results.remove(node.id)
                                    continue
                            except:
                                None
                    nd_start, dst_start, nd_end, dst_end = get_distance_to_closest_split_nodes(way[1], node_idx)
                    dfs(node, None, nd_start, initial_distance - dst_start, 0, set())
                    dfs(node, None, nd_end, initial_distance - dst_end, 0, set())
                else:
                    dfs(node, None, node, initial_distance, 0, set())
                if node.id in results:
                    [output_ways.append(w) for w in tmp]
                tmp = []
                visited_nodes.add(node.id)
                stop = False
    nodes_coords = [v.geometry for k, v in nodes.items() if v.id in results]
    results = filter_close_points_kdtree(nodes_coords, 6)
    res = []
    for way in output_ways:
        tags = ['id', 'highway', 'oneway', 'maxspeed', 'maxspeed:forward', 'maxspeed:backward']
        props = {k: v for k,v in way.tags.items() if k in tags}
        props['id'] = way.id
        res.append(Feature(geometry=LineString(way.geometry), properties=props))

    for coord in results:
        res.append(Feature(geometry=Point(coord), properties={"detection_type": "city_entrance"}))

    feature_collection = FeatureCollection(res)
    with open("/Users/hlebtkach/Downloads/france_visited_ways.geojson", "w") as f:
        json.dump(feature_collection, f)