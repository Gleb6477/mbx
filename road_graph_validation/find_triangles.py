import json
import math
import cv2
import numpy as np
import pandas as pd
import geopy.distance
from geojson import Feature, FeatureCollection
from shapely import LineString, Polygon, Point
from shapely.ops import unary_union, polygonize

ways = {}
results = []
nodes = {}
node_refs = {}
intersection_nodes = {}


def get_distance_in_meters(from_point: np.array, to_point: np.array) -> int:
    from_point, to_point = reversed(from_point), reversed(to_point)
    return geopy.distance.distance(from_point, to_point).m


def parse_features(features):
    for feature in features:
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
            points = 1 if (node_idx == 0 or node_idx == len(feature['nodes']) - 1) else 2
            if node['id'] in intersection_nodes:
                intersection_nodes[node['id']] = intersection_nodes[node['id']] + points
            else:
                intersection_nodes[node['id']] = points

        ways[way_id] = Way(way_id, way_tags, nds, way_coords)

def get_distance(start_node, end_node):
    return geopy.distance.distance(start_node, end_node).m


def has_non_residential_road(sequence, ways_dict):
    non_residential_count = 0
    residential_count = 0

    for way_id in sequence:
        way = ways_dict[way_id]
        if 'highway' in way.tags:
            highway_type = way.tags['highway']
            if highway_type not in  ('residential', 'unclassified'):
                non_residential_count += 1
                if highway_type in ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']:
                    return True
            else:
                residential_count += 1
        else:
            non_residential_count += 1
    return non_residential_count > 0


def is_triangle(geoms, epsilon_factor=0.02, min_area=50.0):
    """
    Check if a collection of LineStrings forms a triangle using OpenCV.

    Steps:
    1. Merge all LineStrings into a single polygon
    2. Convert to image space for OpenCV processing
    3. Use contour approximation to check if polygon has 3 vertices
    4. Apply additional geometric checks

    Parameters:
    - geoms: list of LineString geometries
    - sequence: list of way IDs
    - ways_dict: dictionary of way objects
    - epsilon_factor: factor for contour approximation (higher = more simplification)
    - min_area: minimum area in square meters to consider (avoid tiny triangles)

    Returns:
    - True if the polygon is a triangle, False otherwise
    """
    if len(geoms) < 3:
        return False

    try:
        merged_lines = unary_union(geoms)
        if merged_lines.is_empty or not merged_lines.is_valid:
            return False

        if merged_lines.geom_type == 'LineString':
            if merged_lines.is_closed:
                polygon = Polygon(merged_lines.coords)
            else:
                return False
        elif merged_lines.geom_type == 'MultiLineString':
            polygons = list(polygonize(merged_lines))
            if not polygons:
                return False
            polygon = polygons[0]
        elif merged_lines.geom_type == 'Polygon':
            polygon = merged_lines
        else:
            return False

        if not polygon.is_valid or polygon.is_empty:
            return False
        coords = list(polygon.exterior.coords)

        if len(coords) < 4:
            return False

        def calculate_polygon_area(coords):
            area = 0.0
            n = len(coords)
            for i in range(n):
                j = (i + 1) % n
                area += coords[i][0] * coords[j][1]
                area -= coords[j][0] * coords[i][1]
            area = abs(area) / 2.0

            # This is a rough approximation (1 degree ≈ 111,111 meters at equator)
            avg_lat = sum(coord[1] for coord in coords) / len(coords)
            meters_per_degree_lat = 111111.0
            meters_per_degree_lon = 111111.0 * math.cos(math.radians(avg_lat))
            area_sq_meters = area * meters_per_degree_lat * meters_per_degree_lon
            return area_sq_meters

        area_sq_meters = calculate_polygon_area(coords)

        if area_sq_meters < min_area:
            return False

        min_x = min(coord[0] for coord in coords)
        max_x = max(coord[0] for coord in coords)
        min_y = min(coord[1] for coord in coords)
        max_y = max(coord[1] for coord in coords)

        margin = 0.1
        width = max_x - min_x
        height = max_y - min_y
        min_x -= width * margin
        max_x += width * margin
        min_y -= height * margin
        max_y += height * margin

        # Image dimensions
        img_size = 500  # Use 500x500 image for processing

        # Scale factor
        scale_x = img_size / (max_x - min_x) if (max_x - min_x) > 0 else 1
        scale_y = img_size / (max_y - min_y) if (max_y - min_y) > 0 else 1

        # Convert coordinates to image space
        img_coords = []
        for lon, lat in coords:
            x = int((lon - min_x) * scale_x)
            y = int((lat - min_y) * scale_y)
            # Flip y-axis (image coordinates are top-left origin)
            y = img_size - y
            img_coords.append([x, y])


        image = np.zeros((img_size, img_size), dtype=np.uint8)
        contour = np.array(img_coords, dtype=np.int32)
        cv2.fillPoly(image, [contour], color=255)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        contour = max(contours, key=cv2.contourArea)

        #Approximate contour to polygon
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if we have exactly 3 vertices (triangle)
        if len(approx) == 3:
            vertices = approx.reshape(-1, 2)
            x1, y1 = vertices[0]
            x2, y2 = vertices[1]
            x3, y3 = vertices[2]
            area_img = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
            if area_img < 100:
                return False
            def calculate_angle(a, b, c):
                ba = np.array([a[0] - b[0], a[1] - b[1]])
                bc = np.array([c[0] - b[0], c[1] - b[1]])

                dot_product = np.dot(ba, bc)
                norm_ba = np.linalg.norm(ba)
                norm_bc = np.linalg.norm(bc)

                if norm_ba == 0 or norm_bc == 0:
                    return 0

                cos_angle = dot_product / (norm_ba * norm_bc)
                cos_angle = max(-1.0, min(1.0, cos_angle))
                return math.degrees(math.acos(cos_angle))

            angle1 = calculate_angle(vertices[1], vertices[0], vertices[2])
            angle2 = calculate_angle(vertices[0], vertices[1], vertices[2])
            angle3 = calculate_angle(vertices[0], vertices[2], vertices[1])
            angle_sum = angle1 + angle2 + angle3

            # Allow some tolerance
            if not (160 < angle_sum < 200):
                return False
            if angle1 < 5 or angle2 < 5 or angle3 < 5:
                return False
            if angle1 > 175 or angle2 > 175 or angle3 > 175:
                return False
            return True
        return False

    except Exception as e:
        print(f"Error in OpenCV triangle detection: {e}")
        return False


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
        return self.nodes[len(self.nodes) - 1].id == id


class Node:
    def __init__(self, id, geometry=None, tags=None):
        self.id = id
        self.tags = tags
        self.geometry = geometry


def dfs(initial_node, current_node, distance_left, visited, level, current_path):
    global results
    # Base case: if we've returned to the starting point and have visited at least one way
    if current_node.id == initial_node.id and level > 0 and len(current_path) > 0:
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


if __name__ == "__main__":
    roads = pd.read_csv("/Users/hlebtkach/Downloads/spain_data_for_triangles.csv")
    output_ways = []
    tmp = []
    visited_nodes = set()
    for idx, row in roads.iterrows():
        parse_features(json.loads(row.features))

    for way in ways.items():
        if ('junction' in way[1].tags and (
                way[1].tags['junction'] == 'roundabout' or way[1].tags['junction'] == 'circular')) \
                or ('oneway' in way[1].tags.keys() and way[1].tags['oneway'] in ('yes', '-1', '1')):
            continue
        node = way[1].get_nodes()[0]
        initial_distance = 200.0
        visited = set()
        visited.add(way[0])
        find_closed_geometries(node, initial_distance)

    output = []
    print(f"Total closed geometries found: {len(results)}")

    # Remove duplicates
    unique_tuples = {tuple(sorted(sublist)) for sublist in results}
    unique_lists = [list(tup) for tup in unique_tuples]

    print(f"Unique closed geometries: {len(unique_lists)}")

    triangles_count = 0
    roundabouts_count = 0
    other_count = 0
    residential_only_filtered = 0

    for seq_idx, sequence in enumerate(unique_lists):
        geoms = [LineString(ways[way_id].geometry) for way_id in sequence]
        has_non_residential = has_non_residential_road(sequence, ways)
        if not has_non_residential:
            residential_only_filtered += 1
            continue
        is_triangle = is_triangle(
            geoms,
            epsilon_factor=0.02,
            min_area=50.0
        )
        if is_triangle:
            triangles_count += 1
            print(f"Found triangle #{triangles_count} (sequence {seq_idx + 1}/{len(unique_lists)}): {sequence}")

            highway_types = []
            for way_id in sequence:
                way = ways[way_id]
                if 'highway' in way.tags:
                    highway_types.append(way.tags['highway'])
                else:
                    highway_types.append('no_highway_tag')
            print(f"  Highway types: {highway_types}")
            for way_id in sequence:
                way = ways[way_id]
                props = {k: v for k, v in way.tags.items()}
                props['triangle_id'] = triangles_count
                props['is_triangle'] = True
                props['triangle_sequence'] = seq_idx + 1
                props['has_non_residential'] = has_non_residential
                output.append(Feature(geometry=LineString(way.geometry), properties=props))
    print(f"\n--- Summary ---")
    print(f"Total closed geometries: {len(unique_lists)}")
    print(f"Residential-only geometries filtered: {residential_only_filtered}")
    print(f"Triangles found (with non-residential roads): {triangles_count}")
    print(f"Other closed geometries: {other_count}")
    if triangles_count > 0:
        print(f"\nTriangle statistics:")
        print(f"  - {triangles_count} triangles found with at least one non-residential road")
        print(f"  - {residential_only_filtered} triangles filtered (all residential roads)")
    feature_collection = FeatureCollection(output)
    with open("/Users/hlebtkach/Downloads/spain_triangles.geojson", "w") as f:
        json.dump(feature_collection, f)

    print(f"\nResults saved to: /Users/hlebtkach/Downloads/spain_triangles.geojson")