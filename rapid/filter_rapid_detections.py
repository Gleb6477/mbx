import ast
import hashlib
import json
from json import dump

import geopandas as gpd
import geopy.distance
import numpy as np
import pandas as pd
from geojson import Feature, FeatureCollection
from pyproj import Geod
from shapely import LineString, Point, MultiPolygon, wkt, convex_hull, line_merge, MultiLineString
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
from shapely.ops import polygonize, unary_union
from shapely.prepared import prep
from rtree import index
from shapely.geometry import LineString, MultiLineString

covers_distance = 0.00007
to_polygon = 0.000001
#covers_distance = 0.0073

def remove_all_holes(geometry):
    """Remove all inner holes from a Polygon or MultiPolygon"""
    if isinstance(geometry, Polygon):
        return Polygon(geometry.exterior)
    elif isinstance(geometry, MultiPolygon):
        return MultiPolygon([Polygon(poly.exterior) for poly in geometry.geoms])
    return geometry


def group_intersecting_linestrings(linestrings):
    """
    Simple and robust implementation with error handling
    """
    if not linestrings:
        return []

    # Filter and validate input
    valid_lines = []
    for item in linestrings:
        if isinstance(item, LineString):
            valid_lines.append(item)
        else:
            print(f"Warning: Skipping non-LineString object: {type(item)}")

    if not valid_lines:
        return []

    groups = []
    used_indices = set()

    for i, line in enumerate(valid_lines):
        if i in used_indices:
            continue

        current_group = [line]
        used_indices.add(i)

        found_new = True
        while found_new:
            found_new = False
            for j, other_line in enumerate(valid_lines):
                if j in used_indices:
                    continue
                try:
                    intersects_any = False
                    for line_in_group in current_group:
                        if line_in_group.intersects(other_line):
                            intersects_any = True
                            break

                    if intersects_any:
                        current_group.append(other_line)
                        used_indices.add(j)
                        found_new = True
                except Exception as e:
                    print(f"Warning: Could not check intersection: {e}")
                    continue
        groups.append(current_group)
    result = []
    for group in groups:
        if len(group) == 1:
            result.append(group[0])
        else:
            result.append(MultiLineString(group))

    return result

def remove_all_holes(geometry):
    """Remove all inner holes from a Polygon or MultiPolygon"""
    if isinstance(geometry, Polygon):
        return Polygon(geometry.exterior)
    elif isinstance(geometry, MultiPolygon):
        return MultiPolygon([Polygon(poly.exterior) for poly in geometry.geoms])
    return geometry


def convert_geometry_to_shapely_linestring(dfr):
    geoms = dfr['geometry'].apply(str)
    rowIndex = 0
    for geom in geoms:
        tmp = []
        coordinates = ast.literal_eval(geom.split("{type=LineString, coordinates=")[1][:-1])
        print(str(rowIndex) + "/" + str(len(dfr)))
        for i in range(len(coordinates)):
            tmp.append([coordinates[i][0], coordinates[i][1]])
        #dfr.loc[dfr.index[rowIndex], 'geometry'] = LineString(tmp)
        dfr.loc[dfr.index[rowIndex], 'geometry'] = LineString(tmp).buffer(covers_distance, cap_style=2)
        rowIndex += 1
    return dfr

def convert_geojson_geometry_to_shapely_linestring(dfr):
    # initial_geom = dfr['geometry']
    # dfr['geometry'] = dfr['geometry'].str.replace('LINESTRING (', '')
    # dfr['geometry'] = dfr['geometry'].str.replace(')', '')
    rowIndex = 0
    # geoms = dfr['geometry'].apply(str)
    dfr['geometry'] = dfr['geometry'].apply(wkt.loads)
    for geom in dfr['geometry']:
        # if check_split_points(geom):
        #     dfr.loc[dfr.index[rowIndex], 'point_radius_check'] = True
        print(str(rowIndex) + "/" + str(len(dfr)))
        dfr.loc[dfr.index[rowIndex], 'first_buffered_point'] = Point(geom.coords[0]).buffer(covers_distance-0.000025)
        dfr.loc[dfr.index[rowIndex], 'last_buffered_point'] = Point(geom.coords[len(geom.coords)-1]).buffer(covers_distance-0.000025)
        # first_buffered_point =  Point(geom.coords[0]).buffer(covers_distance)
        # last_buffered_point = Point(geom.coords[len(geom.coords)-1]).buffer(covers_distance)
        # if unified_polygon.intersects(first_buffered_point):
        #     if unified_polygon.intersects(last_buffered_point):
        #         dfr.loc[dfr.index[rowIndex], 'point_radius_check'] = True
        rowIndex += 1
    return dfr

def is_linestring_straight(linestring):
    if not isinstance(linestring, LineString):
        raise TypeError("Input must be a Shapely LineString object.")

    if len(linestring.coords) < 2:
        return True

    start_point = Point(linestring.coords[0])
    end_point = Point(linestring.coords[-1])

    line_length = linestring.length
    tolerance = line_length * 0.2
    straight_distance = start_point.distance(end_point)

    return abs(line_length - straight_distance) < tolerance

features = []

#rapid = gpd.read_file('/Users/hlebtkach/Downloads/test/13122303030.geojson')

gdf = gpd.read_file('/Users/hlebtkach/Downloads/rapid/133002110.geojson')
gdf.to_csv('/Users/hlebtkach/Downloads/rapid/133002110.csv', index=False)

rapid = pd.read_csv('/Users/hlebtkach/Downloads/rapid/133002110.csv')

osm = pd.read_csv('/Users/hlebtkach/Downloads/133002110_osm.csv')

farmlands = pd.read_csv('/Users/hlebtkach/Downloads/133002110_farms.csv')

#farmlands['geometry'] = farmlands['geometry'].replace(loads(farmlands['geometry']))

farmlands['geometry'] = farmlands['geometry'].apply(wkt.loads)
farmlands['geometry'] = farmlands['geometry'].apply(Polygon)

polygons = []
for feature in farmlands['geometry']:
    geom = shape(feature)
    if isinstance(geom, MultiPolygon):
        polygons.extend(list(geom.geoms))
    elif isinstance(geom, Polygon):
        polygons.append(geom)
# 3. Merge polygons
buffer_distance = 0.00075
valid_polygons = [geom.buffer(0) for geom in polygons]
buffered_polygons = [geom.buffer(buffer_distance) for geom in valid_polygons]
merged_geometry = unary_union(buffered_polygons)
final_geometry = merged_geometry.buffer(-buffer_distance + 0.0003)
# 4. Remove holes
solid_farms_geometry = remove_all_holes(final_geometry)

print("converting osm geometries")
osm_filtered_partially = osm[~osm['tags'].str.contains('waterway', regex=True)]
osm_filtered = osm[~osm['tags'].str.contains('waterway|rail|highway=service|highway=track', regex=True)]
osm_buffered = convert_geometry_to_shapely_linestring(osm_filtered)

gdf = gpd.GeoDataFrame(osm_buffered, crs="EPSG:4326")
print("polygons union")
unified_polygon = gdf.geometry.unary_union

points_check_df = osm[~osm['tags'].str.contains('waterway|rail|highway=unclassified|highway=residential|highway=service|highway=track', regex=True)]
points_check_df = convert_geometry_to_shapely_linestring(points_check_df)
gdf = gpd.GeoDataFrame(points_check_df, crs="EPSG:4326")
print("polygons union")
point_check_unified_polygon = gdf.geometry.unary_union

#or 'railway' in osm['tags']
#water_rail_line_df = osm_filtered_partially[osm_filtered_partially['tags'].apply(lambda x : 'railway' in x)]
#gdf = gpd.GeoDataFrame(water_rail_line_df, crs="EPSG:4326")
#water_rail_line = gdf.geometry.unary_union

#farmland_polygon_df = osm_filtered[osm_filtered['tags'].apply(lambda x : 'landuse' in x and x['landuse'] == 'farmland')]
gdf = gpd.GeoDataFrame(farmlands, crs="EPSG:4326")
gdf = gdf.loc[gdf.geometry.is_valid]
farmland_polygon = gdf.geometry.unary_union
# for polygon in solid_farms_geometry.geoms:
#     #individual_polygons.append(polygon)
#     features.append(Feature(geometry=polygon, properties={"farm": True}))
print(farmland_polygon.wkt)
print("farmland area: " + str(farmland_polygon.area))

#individual_polygons = []
#for polygon in unified_polygon.geoms:

    #features.append(Feature(geometry=polygon, properties={"polygon": True}))
polygons_geometry = remove_all_holes(unified_polygon)


print("converting rapid geometries")
rapid['point_radius_check'] = False
rapid = convert_geojson_geometry_to_shapely_linestring(rapid)

# detections with < 10 m length
# detections that are covered by buffered road graph
# detections where first, last points simultaneously do not have road nearby
# detections where there are no any intersections with road graph
# detections with rails intersections
# filter roads inside farmlands
# TODO filter roads where there is at least one point inside parking but not all (geometry points)
# TODO filter intersections with waterways

prepared_unified_polygon = prep(point_check_unified_polygon)

rapid = rapid[rapid.apply(
    lambda r: prepared_unified_polygon.intersects(r['first_buffered_point'])
              or prepared_unified_polygon.intersects(r['last_buffered_point']), axis=1)]

linestrings = []

for index, row in rapid.iterrows():
    geod = Geod(ellps="WGS84")
    # if water_rail_line.intersects(row['geometry']):
    #     props["rail"] = True
    #     print("skipping road due to rail check " + str(index))
    #     continue
    segment_length = geod.geometry_length(row['geometry'])
    intersection_with_farmland = row['geometry'].intersection(farmland_polygon)
    intersection_with_farmland_length = geod.geometry_length(intersection_with_farmland)
    if row['geometry'].within(solid_farms_geometry):
        #props["farm"] = True
        print("skipping road due to farmland check " + str(index))
        continue
    if segment_length < 15:
        print("skipping road due to length check " + str(index))
        continue
    # if not unified_polygon.intersects(row['geometry']):
    #     continue
    intersection_with_graph = row['geometry'].intersection(unified_polygon)
    intersection_with_graph_length = geod.geometry_length(intersection_with_graph)
    if intersection_with_graph_length/segment_length >= min(0.85, segment_length/40 * 0.85):
        #props["overlap"] = True
        print("skipping road due to overlap_check " + str(index))
        continue
    linestrings.append(row['geometry'])

out = group_intersecting_linestrings(linestrings)
for geom in out:
    hash_object = hashlib.sha256(str(geom).encode())
    hex_dig = hash_object.hexdigest()
    props = {"id": int(hex_dig[:8], 16)}
    features.append(Feature(geometry=geom, properties=props))

merged_geometry = unary_union(linestrings)
final_multilinestring = line_merge(merged_geometry)
if isinstance(final_multilinestring, MultiLineString):
    for i, line in enumerate(final_multilinestring.geoms):
        features.append(Feature(geometry=line, properties={}))
        print(f"Component LineString {i}: {line}")
elif isinstance(final_multilinestring, LineString):
    print(f"Result is a single LineString: {final_multilinestring}")
    features.append(Feature(geometry=final_multilinestring, properties={}))


print(len(features))
print(len(out))
feature_collection = FeatureCollection(features)
with open("/Users/hlebtkach/Downloads/133002110.geojson", "w") as f:
    dump(feature_collection, f)
