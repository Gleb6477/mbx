import json
import math

import geopy.distance
import numpy as np
import pandas as pd
from geojson import Feature, FeatureCollection

from shapely import Point, LineString
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union, polygonize

ways = {}
results = []
nodes = {}
node_refs = {}
intersection_nodes = {}

def get_distance_in_meters(from_point: np.array, to_point: np.array) -> int:
    from_point, to_point = reversed(from_point), reversed(to_point)
    return geopy.distance.distance(from_point, to_point).m


def parse_features(feature):
    way_tags = feature['properties']
    way_coords = feature['geometry']['coordinates']
    way_id = feature['id']
    nds = []
    for node_idx in range(0, len(feature['nodes'])):
        node = feature['nodes'][node_idx]
        node_tags = node['properties']
        node_coords = node['geometry']['coordinates']
        node_id = node['id']
        node_obj = Node(node_id, node_coords, node_tags)
        nodes[node_id] = node_obj
        if node_id in node_refs:
            node_refs[node_id].append(feature['id'])
        else:
            node_refs[node_id] = [feature['id']]
        nds.append(node_obj)
        points = 1 if (node_idx == 0 or node_idx == len(feature['nodes'])-1) else 2
        if node['id'] in intersection_nodes:
            intersection_nodes[node['id']] = intersection_nodes[node['id']] + points
        else:
            intersection_nodes[node['id']] = points

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

def dfs(initial_node, current_node, distance_left, visited, level, current_path):
    global results
    # Base case: if we've returned to the starting point and have visited at least one way
    if current_node.id == initial_node.id and level > 0 and len(current_path) > 0:
        #print(f'Found closed geometry at depth: {level}')
        results.append(list(current_path))
        return
    adjacent_ways = node_refs.get(current_node.id, [])
    for way_id in adjacent_ways:
        if way_id in visited:
            continue
        # Skip junctions and oneway roads
        way = ways[way_id]
        if ('junction' in way.tags.keys() or
            ('oneway' in way.tags.keys() and way.tags['oneway'] in ('yes', '-1', '1'))):
            continue
        nodes = way.get_nodes()
        start = nodes[0]
        end = nodes[-1]
        way_length = get_distance(start.geometry, end.geometry)
        if way_length > distance_left:
            continue
        if current_node.id == start.id:
            next_node = end
        elif current_node.id == end.id:
            next_node = start
        else:
            continue
        visited.add(way_id)
        current_path.append(way_id)
        dfs(initial_node, next_node, distance_left - way_length, visited, level + 1, current_path)
        visited.remove(way_id)
        current_path.pop()

def find_closed_geometries(start_node, max_distance):
    visited = set()
    current_path = []
    dfs(start_node, start_node, max_distance, visited, 0, current_path)


def polygon_is_round(geoms):
    """
    Check if a collection of LineStrings forms a round/circular shape
    by calculating centroid from all points and checking distance consistency
    """
    if not geoms or len(geoms) == 0:
        return False

    # Collect all points from all LineStrings
    all_points = []
    for geom in geoms:
        if hasattr(geom, 'coords'):
            all_points.extend(list(geom.coords))
        else:
            # Handle case where it might be a LineString object
            all_points.extend(list(geom.coords))

    if len(all_points) < 3:
        return False

    # Calculate centroid from all points
    x_coords = [point[0] for point in all_points]
    y_coords = [point[1] for point in all_points]
    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)
    #centroid = Point([centroid_x, centroid_y])

    # Calculate distances from centroid to all points
    distances = []
    for point_coords in all_points:
        #point = Point(point_coords)
        distance = math.dist([centroid_x, centroid_y], point_coords)
        distances.append(distance)

    if len(distances) < 2:
        return False

    sorted_distances = sorted(distances)
    min_distance = sorted_distances[0]
    max_distance = sorted_distances[-1]

    if max_distance > min_distance * 1.8:
        return False

    if len(all_points) < 8:
        return False
    return True

def exists_check(nodes):
    exists = 0
    unique_ids = set()
    for nd in nodes:
        if nd.id in unique_ids:
            continue
        else:
            unique_ids.add(nd.id)
        if intersection_nodes[nd.id] >= 3:
            print(nd.id)
            exists += 1
    return exists > 1

if __name__ == "__main__":
    roads = pd.read_csv("/Users/hlebtkach/Downloads/spain.csv")
    output_ways = []
    tmp = []
    visited_nodes = set()
    for idx, row in roads.iterrows():
        parse_features(json.loads(row.feature))

    for way in ways.items():
        if ('junction' in way[1].tags and (way[1].tags['junction'] == 'roundabout' or way[1].tags['junction'] == 'circular'))\
                or ('oneway' in way[1].tags.keys() and way[1].tags['oneway'] in ('yes', '-1', '1')):
            continue
        node = way[1].get_nodes()[0]
        initial_distance = 200.0
        visited = set()
        visited.add(way[0])
        find_closed_geometries(node, initial_distance)

    output = []
    print(len(results))
    roundabouts = 0
    unique_tuples = {tuple(sublist) for sublist in results}
    unique_lists = [list(tup) for tup in unique_tuples]
    for sequence in unique_lists:
        geoms = [LineString(ways[way_id].geometry) for way_id in sequence]
        sequence_nodes_arrays = [ways[way_id].get_nodes() for way_id in sequence]
        sequence_nodes_array = [item for sublist in sequence_nodes_arrays for item in sublist]
        if exists_check(sequence_nodes_array):
            if polygon_is_round(geoms):
                roundabouts += 1
                for way_id in sequence:
                    way = ways[way_id]
                    props = {k: v for k, v in way.tags.items()}
                    output.append(Feature(geometry=LineString(way.geometry), properties=props))
            else:
                None
    print(roundabouts)
    feature_collection = FeatureCollection(output)
    with open("/Users/hlebtkach/Downloads/spain.geojson", "w") as f:
        json.dump(feature_collection, f)