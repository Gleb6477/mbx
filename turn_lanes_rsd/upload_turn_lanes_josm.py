import os
from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas
import pandas as pd
import pyproj
import shapely
import pygeohash as gh

from geopy.distance import geodesic
from geojson import Feature, FeatureCollection, dump, Point, LineString as g_LineString
from geographiclib.geodesic import Geodesic
from shapely import box, LineString, distance, Point, line_locate_point
from shapely.geometry.polygon import Polygon
from shapely.io import from_wkt
import geopandas as gpd
from shapely.ops import nearest_points, snap
from sklearn.neighbors import BallTree
from scripts.graph_traversal.validate_city_entrances import Way, Node

from athena_data_downloader import AthenaDataDownloader
from rave_tools_links import with_google_link

from ways_sl_map import get_quadkey_children, get_coordinates

geod = Geodesic.WGS84
from rtree import index as spatial_index
import warnings
from shapely.geometry import Point, LineString
import math


warnings.filterwarnings("ignore")

proj = pyproj.Transformer.from_crs(3857, 4326, always_xy=True)

def get_query_enriched_ways(quadkey, country, dt):
    QUERY_WAYS = f"""select id, tags, ST_GeometryFromText(geometry_wkt) as geometry
            from roads_production.road_pipeline_corrections_v1_1_0_metadata_corrected_osm_roads_split_enriched where
                metadata_str['iso:3166_1'] = '{country}' and
                dt = '{dt}' and
                cardinality(filter(metadata_array['quadkey_z11'], x -> x in ('{quadkey}')))>0
                and tags['highway'] in ('motorway', 'trunk', 'motorway_link', 'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'unclassified', 'residential', 'tertiary_link', 'service')""".format(
        quadkey=quadkey, dt=dt, country=country
    )
    return QUERY_WAYS

def get_query_signs(quadkey, country):
    lon_min, lat_min, lon_max, lat_max = get_coordinates(quadkey)
    QUERY_SIGNS = f"""
    SELECT
      id,
      lat,
      lon,
      angle,
      'turn_lane' AS feature_type,
    CASE
        WHEN detection_type = 'u_turn_left' then 'reverse_left'
        WHEN detection_type = 'u_turn_right' then 'right_reverse'
        WHEN detection_type = 'straight' then 'through'
        WHEN detection_type = 'straight_right' then 'through_right'
        WHEN detection_type = 'straight_left_right' then 'left_through_right'
        WHEN detection_type = 'straight_left' then 'left_through'
        ELSE detection_type
    END AS arrow_type,
      country AS country_code
    FROM roads_rsd_production.source_v3_0_0_deduplicated_points where country in ('{country}') and
       dt = '2025-11-11' and
       lon > {lon_min} and
       lat > {lat_min} and
       lon < {lon_max} and
       lat < {lat_max} and
       cluster_size > 10 and confidence > 0.1 and status <> 'Deleted' and message_type = 'arrow'
""".format(
       lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max, country=country
    )
    return QUERY_SIGNS

# def get_rsd_traces(quadkey, country, from_dt, to_dt):
#     f"""
#         SELECT
#             id as trace_id,
#             points as coordinates
#         FROM {rsd_db}.{traces_table}
#         WHERE dt BETWEEN '{from_dt}' AND '{to_dt}'
#         AND SIZE(FILTER(qks, x -> RLIKE(x, '{quadkey}'))) > 0
#         AND country IN ({country})
#     """

prolongate_from = []

def point(lon: float, lat: float, hdg: float, properties: dict, length: float = 5) -> Feature:
    return Feature(geometry=Point((float(lon), float(lat))), properties=properties)

def prepare_rsd_points(data: pandas.DataFrame):
    all_data = []
    for id, lat, lon, angle, arrow_type in zip(
        data["id"],
        data["lat"],
        data["lon"],
        data["angle"],
        data["arrow_type"]):
        value = 0
        all_data.append(
            point(
                lon,
                lat,
                angle,
                properties={
                    "id": id,
                    "angle": angle,
                    "detection_type": arrow_type
                },
            )
        )
    return all_data

basic_tags = ['id', 'highway', 'name', 'oneway']
prolongation_tags = ['angle', 'prolongation_part', 'perpendicular_distance']

def df_to_geojson(df):
    features = []
    for idx, row in df.iterrows():
        props = row.to_dict()
        del props['geometry']
        non_empty = {key: value for key, value in props.items() if not pd.isna(value)}
        filtered_props = {k:v for k,v in non_empty.items()
                          if k in basic_tags or k in prolongation_tags or ('lanes' in k or 'turn' in k)
                          and 'correction' not in k
                          and 'source' not in k
                          and 'parking' not in k}
        features.append(Feature(geometry=row['geometry'], properties=filtered_props))
    return features

def get_osm_download_zone(qk_list):
    polygon = Polygon()
    for qk in qk_list:
        qk_bounds = get_coordinates(qk)
        bbox = box(*qk_bounds)
        polygon = polygon.union(Polygon(bbox))
    return polygon.__str__()

def parse_df_tags(df):
    for idx, row in df.iterrows():
        props = {'id': row.id}
        for s_i, s_v in row.items():
            if s_i == 'tags':
                s_v = s_v.strip("{}")
                tmp_pair = s_v.split(", ")
                for pair in tmp_pair:
                    try:
                        key, value = pair.split("=")
                    except:
                        continue
                    if value:
                        props[key] = value

        props = {key: '0' if value == '[DELETED]' else value for key, value in props.items()}
        if 'oneway' not in props.keys():
            props['oneway'] = 'no'
        for tag in props.items():
            df.loc[df.index[idx], tag[0]] = tag[1]
    df = df.drop(['tags'], axis=1)
    return df

ways = {}
nodes = {}
node_refs = {}

def hash_node_by_lat_lon(latitude, longitude, precision=12):
    try:
        geohash_string = gh.encode(latitude, longitude, precision=precision)
        return geohash_string
    except Exception as e:
        print(f"Error encoding Geohash: {e}")
        return None

maneuvers = {}

def prepare_features_for_traversal(roads_df):
    ways.clear()
    nodes.clear()
    node_refs.clear()
    for _, road in roads_df.iterrows():
        way_coords = list(road['geometry'].coords)
        way_id = road['id']

        if way_id in ways:
            continue

        nds = []
        for node_idx in range(0, len(way_coords)):
            node_coords = way_coords[node_idx]
            node_id = hash_node_by_lat_lon(node_coords[1], node_coords[0])
            node_obj = Node(node_id, node_coords)
            nodes[node_id] = node_obj
            iterations = 2 if (0 < node_idx < len(way_coords) - 1) else 1

            for it in range(0, iterations):
                if node_id in node_refs:
                    if way_id not in node_refs[node_id]:
                        node_refs[node_id].append(way_id)
                else:
                    node_refs[node_id] = [way_id]
            nds.append(node_obj)

            # Exclude incoming oneways (impossible maneuver)
            # if node_idx == len(way_coords) - 1 and road['oneway'] in ('yes', '-1'):
            #     if node_id in maneuvers:
            #         maneuvers[node_id].append(way_id)
            #     else:
            #         maneuvers[node_id] = [way_id]

        ways[way_id] = Way(way_id, None, nds, way_coords)


united_roads = []

def get_intersection_to_intersection_array_from_road():
    for candidate in prolongate_from:
        road_id = candidate['road_attributes']['id_right']
        visited = set()
        united_roads.append(road_id)
        start = candidate['prolongation_start_point_id']
        start_node = nodes[start]
        all_data.append(Feature(geometry=Point(start_node.geometry),
                              properties={'start': start, 'num_of_nodes': len(ways[road_id].nodes)}))
        visited.add(road_id)
        dfs(start, visited)
exclude_from_united_roads = []

def dfs(node_id, visited):
    neighbors = node_refs[node_id]
    if len(neighbors) >= 3:
        for n in neighbors:
            visited.add(n)
        return
    for neighbor in neighbors:
        if neighbor in visited:
            continue
        united_roads.append(neighbor)
        visited.add(neighbor)
        way_nodes = ways[neighbor].nodes
        for node in way_nodes:
            dfs(node.id, visited)

def calculate_angle_linear_0_360(linestring, projected_point, sign_angle):
    # Project the point onto the linestring
    projection_distance = linestring.project(projected_point)
    closest_point = linestring.interpolate(projection_distance)
    # Get points before and after the projection point
    if projection_distance == 0:
        point_before = Point(linestring.coords[0])
        point_after = Point(linestring.coords[1])
    elif projection_distance >= linestring.length:
        point_before = Point(linestring.coords[-2])
        point_after = Point(linestring.coords[-1])
    else:
        accumulated = 0
        coords = linestring.coords
        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i + 1]])
            segment_length = segment.length
            if accumulated <= projection_distance <= accumulated + segment_length:
                point_before = Point(coords[i])
                point_after = Point(coords[i + 1])
                break
            accumulated += segment_length

    before = Point(proj.transform(point_before.x, point_before.y))
    after = Point(proj.transform(point_after.x, point_after.y))
    closest_point = Point(proj.transform(closest_point.x, closest_point.y))
    #print('before prolongation index')
    prolongation_start_idx = (
        0) if (angle_difference(get_rotation_angle(closest_point, before), sign_angle)
                    < angle_difference(get_rotation_angle(closest_point, after), sign_angle))\
        else (len(linestring.coords) - 1)
    get_rotation_angle(closest_point, before)
    get_rotation_angle(closest_point, after)

    # Calculate vectors
    x1, y1 = point_before.x, point_before.y
    x2, y2 = point_after.x, point_after.y

    dx = x2 - x1
    dy = y2 - y1

    angle_radians = math.atan2(dx, dy)
    angle_degrees = math.degrees(angle_radians)
    if angle_degrees < 0:
        angle_degrees += 360
    return angle_degrees, prolongation_start_idx

def calculate_distance_epsg3857(point1, point2):
    x1, y1 = point1.x, point1.y
    x2, y2 = point2.x, point2.y
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_match_score(distance, angle_diff, alpha=0.005, beta=1, angle_threshold=15.0, max_angle=180.0):
    normalized_angle = min(abs(angle_diff) / max_angle, 1.0)
    if abs(angle_diff) <= angle_threshold:
        scaled_angle = math.sqrt(normalized_angle) * (angle_threshold / max_angle)
    else:
        scaled_angle = normalized_angle

    angle_multiplier = 1
    if angle_diff > 30:
        angle_multiplier = (max_angle + angle_diff) / max_angle
    angle_term = scaled_angle / (math.pow(alpha, angle_multiplier) * distance)
    distance_term = beta * distance
    score = angle_term + distance_term
    return score

def get_suitable_candidate(roads, sign_angle):
    sorted_candidates = sorted(roads, key=lambda item: item['perpendicular_distance'])
    for candidate in sorted_candidates:
        if angle_difference(sign_angle, candidate['angle'], candidate['oneway']) <= 30:
            return candidate
    return None

def get_rotation_angle(base_point, comp_point):
    dx = base_point.x - comp_point.x
    dy = base_point.y - comp_point.y
    theta = math.atan2(dx, dy)
    ang = theta*180/math.pi
    while ang < 0:
        ang+=360
    return ang

def angle_difference(angle1, angle2, oneway='no'):
    diff = abs(angle1 - angle2)
    return min(min(diff, 180 - diff), min(diff, 360 - diff)) if oneway=='no' else min(diff, 360 - diff)


def split_way_at_point(way_id, split_point, original_way):
    """Split a way at the closest point and return two new ways"""
    way_geom = original_way['geometry']
    split_distance = way_geom.project(split_point)

    # Split the linestring at the specified distance
    way_coords = list(way_geom.coords)

    # Find the segment where the split occurs
    print(split_point.wkt)
    print(way_geom.wkt)
    accumulated = 0
    split_index = None
    for i in range(len(way_coords) - 1):
        segment = LineString([way_coords[i], way_coords[i + 1]])
        segment_length = segment.length
        if accumulated <= split_distance <= accumulated + segment_length:
            # Calculate the exact split point within this segment
            segment_ratio = (split_distance - accumulated) / segment_length
            split_lon = way_coords[i][0] + segment_ratio * (way_coords[i + 1][0] - way_coords[i][0])
            split_lat = way_coords[i][1] + segment_ratio * (way_coords[i + 1][1] - way_coords[i][1])
            split_coord = (split_lon, split_lat)
            split_index = i + 1
            break
        accumulated += segment_length

    if split_index is None:
        # If no split point found, return original way
        return [original_way], [split_point]

    # Create two new ways
    way1_coords = way_coords[:split_index] + [split_coord]
    way2_coords = [split_coord] + way_coords[split_index:]

    # Create new way objects
    way1_id = f"{way_id}_a"
    way2_id = f"{way_id}_b"

    way1 = original_way.copy()
    way2 = original_way.copy()

    way1['id'] = way1_id
    way2['id'] = way2_id
    way1['geometry'] = LineString(way1_coords)
    way2['geometry'] = LineString(way2_coords)

    return [way1, way2], split_coord


def update_data_structures_after_split(original_way_id, new_ways, split_coord):
    """Update nodes, ways, and node_refs after splitting a way"""
    split_node_id = hash_node_by_lat_lon(split_coord[1], split_coord[0])

    # Remove original way from data structures
    if original_way_id in ways:
        del ways[original_way_id]

    # Remove original way from node_refs
    for node_id, way_list in list(node_refs.items()):
        if original_way_id in way_list:
            node_refs[node_id] = [w for w in way_list if w != original_way_id]
            if not node_refs[node_id]:
                del node_refs[node_id]

    # Add new ways and their nodes
    for new_way in new_ways:
        way_coords = list(new_way['geometry'].coords)
        way_id = new_way['id']
        nds = []

        for node_idx, node_coords in enumerate(way_coords):
            node_id = hash_node_by_lat_lon(node_coords[1], node_coords[0])
            node_obj = Node(node_id, node_coords)
            nodes[node_id] = node_obj
            nds.append(node_obj)

            # Update node_refs
            iterations = 2 if (0 < node_idx < len(way_coords) - 1) else 1
            for _ in range(iterations):
                if node_id in node_refs:
                    if way_id not in node_refs[node_id]:
                        node_refs[node_id].append(way_id)
                else:
                    node_refs[node_id] = [way_id]

        # Create Way object
        ways[way_id] = Way(way_id, None, nds, way_coords)

        # Handle oneway restrictions
        if new_way['oneway'] in ('yes', '-1') and node_idx == len(way_coords) - 1:
            last_node_id = nds[-1].id
            if last_node_id in maneuvers:
                maneuvers[last_node_id].append(way_id)
            else:
                maneuvers[last_node_id] = [way_id]

    return split_node_id


def find_best_segment_after_split(split_ways, sign_angle, split_point):
    """Find the best segment after splitting based on road bearing and sign angle"""
    best_segment = None
    best_angle_diff = float('inf')

    for way in split_ways:
        way_geom = way['geometry']
        way_coords = list(way_geom.coords)

        # Calculate bearing for this segment
        if len(way_coords) >= 2:
            # Determine direction based on which end is closer to split point
            start_point = Point(way_coords[0])
            end_point = Point(way_coords[-1])

            start_dist = distance(start_point, split_point)
            end_dist = distance(end_point, split_point)

            # Use the end farther from split point to determine direction
            if start_dist > end_dist:
                bearing = calculate_bearing(way_coords[0], way_coords[1])
            else:
                bearing = calculate_bearing(way_coords[-1], way_coords[-2])
            angle_diff = angle_difference(sign_angle, bearing, way['oneway'])
            if angle_diff < best_angle_diff:
                best_angle_diff = angle_diff
                best_segment = way

    return best_segment, best_angle_diff


def calculate_bearing(coord1, coord2):
    """Calculate bearing between two coordinates"""
    lat1, lon1 = coord1[1], coord1[0]
    lat2, lon2 = coord2[1], coord2[0]

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon_rad = math.radians(lon2 - lon1)

    x = math.sin(delta_lon_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad)

    bearing_rad = math.atan2(x, y)
    bearing_deg = math.degrees(bearing_rad)

    return (bearing_deg + 360) % 360


def match_points_sjoin_optimized(roads_df, points_df, max_distance=30,
                                 point_geometry_col='geometry', road_geometry_col='geometry'):
    output = []
    points_gdf = gpd.GeoDataFrame(points_df, geometry=point_geometry_col, crs="EPSG:4326")
    roads_gdf = gpd.GeoDataFrame(roads_df, geometry=road_geometry_col, crs="EPSG:4326")

    # EPSG:3857 for metric calculations
    metric_crs = "EPSG:3857"
    points_metric = points_gdf.to_crs(metric_crs)
    roads_metric = roads_gdf.to_crs(metric_crs)

    points_buffered = points_metric.copy()
    points_buffered['geometry'] = points_metric.geometry.buffer(max_distance)

    joined = gpd.sjoin(points_buffered, roads_metric, how='left', predicate='intersects')

    for point_idx in points_metric.index:
        point_candidates = joined[joined.index == point_idx]
        if len(point_candidates) == 0:
            points_df.loc[point_idx, 'id'] = 0
            continue

        candidate_roads = []
        point_geom = points_metric.loc[point_idx].geometry
        sign_angle = points_metric.loc[point_idx].angle

        for _, candidate_row in point_candidates.iterrows():
            try:
                road_geom = roads_metric.loc[candidate_row.index_right].geometry
                original_way_id = roads_metric.loc[candidate_row.index_right].id
            except:
                continue

            # Find closest point on road
            perpendicular_distance = road_geom.project(point_geom)
            closest_point = road_geom.interpolate(perpendicular_distance)
            perpendicular_distance_val = calculate_distance_epsg3857(point_geom, closest_point)

            if perpendicular_distance_val <= max_distance:
                # Convert back to EPSG:4326 for splitting
                closest_point_4326 = Point(proj.transform(closest_point.x, closest_point.y))
                original_road_data = roads_df[roads_df['id'] == original_way_id].iloc[0]

                # Split the way at the closest point
                split_ways, split_coord = split_way_at_point(original_way_id, closest_point_4326, original_road_data)
                # Update data structures with split ways
                split_node_id = update_data_structures_after_split(original_way_id, split_ways, split_coord)

                # Find the best segment after splitting
                best_segment, best_angle_diff = find_best_segment_after_split(split_ways, sign_angle,
                                                                              closest_point_4326)

                if best_segment:
                    # Determine prolongation start point (opposite end from split)
                    segment_coords = list(best_segment['geometry'].coords)
                    start_point = Point(segment_coords[0])
                    end_point = Point(segment_coords[-1])

                    # Use the end farther from split point as prolongation start
                    if distance(start_point, closest_point_4326) > distance(end_point, closest_point_4326):
                        prolongation_start_id = hash_node_by_lat_lon(segment_coords[0][1], segment_coords[0][0])
                    else:
                        prolongation_start_id = hash_node_by_lat_lon(segment_coords[-1][1], segment_coords[-1][0])

                    # Calculate road angle for the best segment
                    road_angle = calculate_bearing(segment_coords[0], segment_coords[1]) if len(
                        segment_coords) >= 2 else 0

                    candidate_roads.append({
                        'point_geom': point_geom,
                        'closest_point': closest_point,
                        'road_id': best_segment['id'],
                        'perpendicular_distance': perpendicular_distance_val,
                        'angle': road_angle,
                        'oneway': best_segment['oneway'],
                        'road_attributes': best_segment.to_dict(),
                        'prolongation_start_point_id': prolongation_start_id,
                        'angle_diff': best_angle_diff
                    })

        if candidate_roads:
            # Sort by angle difference and distance
            candidate_roads.sort(key=lambda x: (x['angle_diff'], x['perpendicular_distance']))
            best_candidate = candidate_roads[0]

            prolongate_from.append(best_candidate)
            point_geom = best_candidate['point_geom']
            closest_point = best_candidate['closest_point']

            points_df.loc[point_idx, 'matched_road_id'] = best_candidate['road_id']
            points_df.loc[point_idx, 'distance'] = best_candidate['perpendicular_distance']

            output.append(Feature(geometry=Point(proj.transform(closest_point.x, closest_point.y)),
                                  properties={'road_angle': best_candidate['angle']}))
            output.append(Feature(
                geometry=g_LineString([list(proj.transform(point_geom.x, point_geom.y)),
                                       list(proj.transform(closest_point.x, closest_point.y))]),
                properties={'perpendicular_distance': best_candidate['perpendicular_distance']}))

    return output

def filter_detections(roads, signs):
    filtered_roads = roads[roads['highway'] != 'service']
    output = []
    points_gdf = gpd.GeoDataFrame(signs, geometry='geometry', crs="EPSG:4326")
    roads_gdf = gpd.GeoDataFrame(filtered_roads, geometry='geometry', crs="EPSG:4326")

    #EPSG:3857
    metric_crs = "EPSG:3857"
    points_metric = points_gdf.to_crs(metric_crs)
    roads_metric = roads_gdf.to_crs(metric_crs)

    roads_buffered = roads_metric.copy()
    roads_buffered['geometry'] = roads_buffered.geometry.buffer(30)

    joined = gpd.sjoin(points_metric, roads_buffered, how='left', predicate='intersects')
    matches = []
    for point_idx in points_metric.index:
        point_candidates = joined[joined.index == point_idx]
        point_candidates = point_candidates[~point_candidates['index_right'].isna()]
        if len(point_candidates) == 0:
            signs.loc[point_idx, 'id'] = 0

if __name__ == "__main__":
    root = '/Users/hlebtkach/upload_tls/'
    # qks = '12020211201,12020211203'
    qks = '1202300020,1202300021,1202300022,1202300023'
    #qks = '02301022111,02301023000,02301022113,02301023002'
    #current_dt = datetime.now().date()
    #country = 'US'
    country = 'DE'
    #folder_name = str(current_dt) + '/' + country
    folder_name = '2025-11-05/DE'
    dt = '2025-11-05'
    #dt = str(current_dt - timedelta(days=2))
    #from_dt = str(current_dt - timedelta(days=5))
    quadkeys = str(qks).split(",")
    print(quadkeys)
    list_qk_11 = []
    for qk in quadkeys:
        list_qk_11.extend(get_quadkey_children(qk, 10))
    qks_to_download = list(set(list_qk_11))
    print("Download qks and signs")
    folder = os.path.join(root, folder_name)
    if not os.path.exists(os.path.join(root, folder_name)):
        os.makedirs(os.path.join(root, folder_name))
    if not os.path.exists(os.path.join(root, folder_name, "to_do")):
        os.makedirs(os.path.join(root, folder_name, "to_do"))

    qks_name = qks.replace(',', '_')
    print(qks_name)
    for qk in qks_to_download:
        # if os.path.exists(os.path.join(args.root, args.folder_name, "to_do", qk+".geojson")):
        #     print(f"{qk} skipped")
        #     continue
        file_mapbox_ways = f"{folder}/{qk}_mapbox.csv".format(folder=folder, qk=qk)
        file_signs = f"{folder}/{qk}_signs.csv".format(folder=folder, qk=qk)
        #file_traces = f"{folder}/{qk}_traces.csv".format(folder=folder, qk=qk)
        out_file = f"{folder}/to_do/{qk}.geojson".format(folder=folder, qk=qk)
        AthenaDataDownloader().download_data(get_query_enriched_ways(qk, country, dt), "road-internal", file_mapbox_ways)
        AthenaDataDownloader().download_data(get_query_signs(qk, country), "road-internal", file_signs)

        qk_df = pd.read_csv(file_mapbox_ways)
        signs_df = pd.read_csv(file_signs)
        signs_df['geometry'] = ''
        signs_df['geometry'] = signs_df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
        qk_df['geometry'] = qk_df['geometry'].apply(lambda x: from_wkt(x))
        qk_df = parse_df_tags(qk_df)
        prepare_features_for_traversal(qk_df)
        #filter_detections(qk_df, signs_df)
        #signs_df = signs_df[signs_df['id'] != 0]
        all_data = match_points_sjoin_optimized(qk_df, signs_df)
        get_intersection_to_intersection_array_from_road()
        print(len(united_roads))
        #result_list = [int(item1) ^ int(item2) for item1, item2 in zip(united_roads, exclude_from_united_roads)]
        #qk_df['prolongation_part'] = False
        #qk_df['prolongation_part'] = qk_df.apply(lambda row: row['id'] in united_roads, axis=1)
        all_data.extend(df_to_geojson(qk_df))

        signs = prepare_rsd_points(signs_df)
        all_data.extend(signs)
        with open(out_file, "w") as f:
            dump(FeatureCollection(all_data), f)