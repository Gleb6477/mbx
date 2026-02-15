import json
from pathlib import Path

import rtree
from shapely.geometry import shape, LineString, Point, mapping
from shapely.ops import transform, linemerge
import pyproj
from functools import partial
from collections import defaultdict, deque
import numpy as np


def load_and_reproject_geojson(filepath, source_crs='EPSG:4326', target_crs='EPSG:3857'):
    with open(filepath, 'r') as f:
        data = json.load(f)

    project = pyproj.Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    ).transform

    unproject = pyproj.Transformer.from_crs(
        target_crs, source_crs, always_xy=True
    ).transform

    features = []
    for feature in data['features']:
        geom_source = shape(feature['geometry'])

        # Convert MultiLineString to list of LineStrings
        if geom_source.geom_type == 'MultiLineString':
            line_strings = list(geom_source.geoms)
        else:
            line_strings = [geom_source]

        for line in line_strings:
            geom_target = transform(project, line)

            # Get the original ID from properties
            props = feature.get('properties', {})
            original_id = props.get('id') or props.get('ID') or props.get('@id') or str(len(features))

            features.append({
                'geometry_source': line,
                'geometry_target': geom_target,
                'properties': feature.get('properties', {}).copy(),
                'original': feature,
                'original_id': original_id,  # Store the original ID
                'feature_index': len(features)  # Index in the features list
            })

    return features, project, unproject


def build_road_graph_with_weighted_intersections(features, tolerance=0.1):
    """
    Build road graph with weighted intersection detection:
    - Endpoint contributes 1
    - Internal point contributes 2
    - Total >= 3 means it's a true intersection


    Rules for merging:
    1. Never merge oneway with non-oneway roads
    2. For oneway roads: if oneway=-1, reverse geometry before analysis
    3. Merge only if endpoints connect properly (end of road A to start of road B)
    """
    print(f"  Building road graph for {len(features)} roads...")

    # Helper function to get oneway status
    def get_oneway_info(feature):
        props = feature.get('properties', {})
        oneway_value = props.get('oneway') or props.get('oneway:') or props.get('Oneway') or props.get('ONEWAY')

        if oneway_value is None:
            return False, False  # Not oneway, not reversed

        oneway_str = str(oneway_value).lower().strip()

        if oneway_str in ['no', 'false', '0', 'bidirectional', 'two-way']:
            return False, False  # Bidirectional, not reversed
        elif oneway_str in ['yes', 'true', '1', 'forward']:
            return True, False  # Oneway forward, not reversed
        elif oneway_str in ['-1', 'reverse', 'backward']:
            return True, True  # Oneway reverse, needs reversal
        else:
            return False, False  # Unknown, treat as bidirectional

    # Helper function to reverse geometry if needed
    def prepare_geometry(feature):
        geom_target = feature['geometry_target']
        geom_source = feature['geometry_source']

        is_oneway, needs_reverse = get_oneway_info(feature)

        if needs_reverse:
            # Reverse target geometry
            coords_target = list(geom_target.coords)
            reversed_coords_target = list(reversed(coords_target))
            geom_target = LineString(reversed_coords_target)

            # Reverse source geometry
            coords_source = list(geom_source.coords)
            reversed_coords_source = list(reversed(coords_source))
            geom_source = LineString(reversed_coords_source)

            # Update feature with reversed geometries
            feature['geometry_target'] = geom_target
            feature['geometry_source'] = geom_source
            feature['_reversed_for_analysis'] = True
        else:
            feature['_reversed_for_analysis'] = False

        feature['_is_oneway'] = is_oneway
        feature['_needs_reverse'] = needs_reverse

        return geom_target

    # Helper function to create node key with tolerance
    def create_node_key(point):
        return (round(point.x / tolerance) * tolerance,
                round(point.y / tolerance) * tolerance)

    # Step 1: Prepare geometries and build node graph
    print("  Preparing geometries and building node graph...")
    node_graph = defaultdict(lambda: {
        'point': None,
        'connection_weight': 0,
        'endpoint_roads': set(),  # set of (road_idx, is_start) tuples
        'internal_roads': set(),
        'road_positions': defaultdict(list)
    })

    # Track oneway status for each road
    road_oneway_status = {}  # road_idx -> (is_oneway, was_reversed)

    # Process all roads
    for road_idx, feature in enumerate(features):
        geom_target = prepare_geometry(feature)

        if geom_target.geom_type != 'LineString':
            continue

        coords = list(geom_target.coords)
        if len(coords) < 2:
            continue

        # Store coordinates for later use
        feature['_coords'] = coords
        feature['_source_coords'] = list(feature['geometry_source'].coords)

        # Store oneway status
        is_oneway = feature['_is_oneway']
        was_reversed = feature['_reversed_for_analysis']
        road_oneway_status[road_idx] = (is_oneway, was_reversed)

        # Process all vertices
        for idx, coord in enumerate(coords):
            point = Point(coord)
            key = create_node_key(point)
            node_graph[key]['point'] = point
            node_graph[key]['road_positions'][road_idx].append(idx)

            # Check if this is an endpoint
            is_endpoint = (idx == 0 or idx == len(coords) - 1)

            if is_endpoint:
                is_start = (idx == 0)
                node_graph[key]['endpoint_roads'].add((road_idx, is_start))
            else:
                node_graph[key]['internal_roads'].add(road_idx)

    # Step 2: Calculate connection weight for each node
    print("  Calculating connection weights...")
    true_intersection_nodes = set()

    for key, node_info in node_graph.items():
        # Calculate connection weight:
        # Each endpoint road contributes 1
        # Each internal road contributes 2
        weight = (len(node_info['endpoint_roads']) * 1 +
                  len(node_info['internal_roads']) * 2)

        node_info['connection_weight'] = weight
        # Node is a true intersection if weight >= 3
        if weight >= 3:
            true_intersection_nodes.add(key)
            node_info['is_true_intersection'] = True
        else:
            node_info['is_true_intersection'] = False

    print(f"  Found {len(true_intersection_nodes)} true intersection nodes (weight >= 3)")

    # Step 3: Find roads to merge based on endpoint connections
    print("  Identifying endpoint connections to merge...")

    # Build connection graph with oneway compatibility check
    connection_graph = defaultdict(set)
    connection_details = {}  # (road1, road2) -> connection type

    for key, node_info in node_graph.items():
        if not node_info['is_true_intersection']:
            endpoint_roads = list(node_info['endpoint_roads'])

            # Check all pairs of endpoint roads at this node
            for i in range(len(endpoint_roads)):
                for j in range(i + 1, len(endpoint_roads)):
                    road1, is_start1 = endpoint_roads[i]
                    road2, is_start2 = endpoint_roads[j]

                    # Get oneway status for both roads
                    is_oneway1, _ = road_oneway_status.get(road1, (False, False))
                    is_oneway2, _ = road_oneway_status.get(road2, (False, False))

                    # Rule 1: Don't merge oneway with non-oneway
                    if is_oneway1 != is_oneway2:
                        continue

                    # For oneway roads: check proper connection
                    if is_oneway1 and is_oneway2:
                        # Oneway roads can only connect end-to-start
                        # end of road1 (not start) should connect to start of road2
                        if (not is_start1) and is_start2:
                            # Good connection: end of road1 to start of road2
                            connection_graph[road1].add(road2)
                            connection_graph[road2].add(road1)
                            connection_details[(road1, road2)] = 'oneway_end_to_start'
                        elif (not is_start2) and is_start1:
                            # Good connection: end of road2 to start of road1
                            connection_graph[road1].add(road2)
                            connection_graph[road2].add(road1)
                            connection_details[(road1, road2)] = 'oneway_end_to_start'
                        # Don't connect if both are starts or both are ends
                    else:
                        # Both are bidirectional - can connect any endpoint to any endpoint
                        connection_graph[road1].add(road2)
                        connection_graph[road2].add(road1)
                        connection_details[(road1, road2)] = 'bidirectional'

    # Step 4: Find connected components for merging
    print("  Finding connected components for merging...")
    visited = set()
    road_groups = []

    for road_idx in range(len(features)):
        if road_idx in visited or '_coords' not in features[road_idx]:
            continue

        # Start a new BFS for this component
        queue = deque([road_idx])
        component = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue

            visited.add(current)
            component.add(current)

            # Add all connected roads (with proper oneway compatibility)
            for neighbor in connection_graph[current]:
                if neighbor not in visited:
                    # Double-check oneway compatibility
                    is_oneway1, _ = road_oneway_status.get(current, (False, False))
                    is_oneway2, _ = road_oneway_status.get(neighbor, (False, False))

                    if is_oneway1 == is_oneway2:
                        queue.append(neighbor)

        if len(component) > 1:
            road_groups.append(list(component))

    print(f"  Found {len(road_groups)} groups of roads to merge")

    # Step 5: Merge roads in each group
    merged_features = []
    merged_map = {}  # original road_idx -> new merged segment index

    # First, handle non-merged roads
    all_roads = set(range(len(features)))
    merged_roads = set()
    for group in road_groups:
        merged_roads.update(group)

    non_merged_roads = all_roads - merged_roads

    for road_idx in non_merged_roads:
        feature = features[road_idx].copy()
        feature['segment_id'] = f"{road_idx}_0"
        feature['parent_id'] = road_idx
        feature['is_merged'] = False
        feature['merged_from'] = [road_idx]
        # Store original ID for this feature
        feature['original_ids'] = [feature['original_id']]  # List of original IDs
        merged_features.append(feature)
        merged_map[road_idx] = len(merged_features) - 1

    # Now merge each group
    for group_idx, group in enumerate(road_groups):
        print(f"    Merging group {group_idx + 1} with {len(group)} roads")

        # Check if group contains oneway roads
        group_is_oneway = False
        for road_idx in group:
            is_oneway, _ = road_oneway_status.get(road_idx, (False, False))
            if is_oneway:
                group_is_oneway = True
                break

        # Collect all line segments from this group
        all_segments = []
        source_segments = []
        group_original_ids = []  # Collect original IDs from all roads in group

        for road_idx in group:
            feature = features[road_idx]
            if '_coords' not in feature:
                continue

            # Add target geometry segment
            geom_target = feature['geometry_target']
            all_segments.append(geom_target)

            # Add source geometry segment
            geom_source = feature['geometry_source']
            source_segments.append(geom_source)

            # Collect original ID
            group_original_ids.append(feature['original_id'])

            # Map original road to this merged segment
            merged_map[road_idx] = len(merged_features)

        # Merge target geometries
        try:
            merged_target = linemerge(all_segments)
            if merged_target.geom_type == 'MultiLineString':
                # Try to simplify further
                lines = list(merged_target.geoms)
                if lines:
                    # Take the longest as base
                    merged_target = max(lines, key=lambda x: x.length)
        except:
            # If linemerge fails, use the longest segment
            longest_idx = max(range(len(all_segments)), key=lambda i: all_segments[i].length)
            merged_target = all_segments[longest_idx]

        # Merge source geometries
        try:
            merged_source = linemerge(source_segments)
            if merged_source.geom_type == 'MultiLineString':
                lines = list(merged_source.geoms)
                if lines:
                    merged_source = max(lines, key=lambda x: x.length)
        except:
            longest_idx = max(range(len(source_segments)), key=lambda i: source_segments[i].length)
            merged_source = source_segments[longest_idx]

        # Create merged feature
        merged_feature = features[group[0]].copy()  # Use first road as template
        merged_feature['geometry_target'] = merged_target
        merged_feature['geometry_source'] = merged_source
        merged_feature['segment_id'] = f"merged_{group_idx}"
        merged_feature['parent_id'] = group[0]
        merged_feature['is_merged'] = True
        merged_feature['merged_from'] = group.copy()
        merged_feature['original_ids'] = group_original_ids  # Store all original IDs
        merged_feature['properties']['merged_count'] = len(group)
        merged_feature['_merged_group_oneway'] = group_is_oneway

        merged_features.append(merged_feature)

    print(f"  After merging: {len(merged_features)} segments")

    # Step 6: Split merged roads at true intersections
    print("  Splitting at true intersections...")
    final_segments = []

    for seg_idx, feature in enumerate(merged_features):
        geom = feature['geometry_target']
        if feature['properties']['id'] == '924166329317558':
            print(1)
        if geom.geom_type != 'LineString':
            segment_feature = feature.copy()
            segment_feature['segment_id'] = f"{seg_idx}_0"
            segment_feature['parent_id'] = seg_idx
            segment_feature['is_split'] = False
            # Keep original IDs from parent
            if 'original_ids' in feature:
                segment_feature['original_ids'] = feature['original_ids'].copy()
            else:
                segment_feature['original_ids'] = [feature.get('original_id', str(seg_idx))]
            final_segments.append(segment_feature)
            continue

        coords = list(geom.coords)
        if len(coords) < 2:
            segment_feature = feature.copy()
            segment_feature['segment_id'] = f"{seg_idx}_0"
            segment_feature['parent_id'] = seg_idx
            segment_feature['is_split'] = False
            # Keep original IDs from parent
            if 'original_ids' in feature:
                segment_feature['original_ids'] = feature['original_ids'].copy()
            else:
                segment_feature['original_ids'] = [feature.get('original_id', str(seg_idx))]
            final_segments.append(segment_feature)
            continue

        # Find true intersection nodes along this road
        split_indices = []

        for idx, coord in enumerate(coords):
            point = Point(coord)
            key = create_node_key(point)

            if key in true_intersection_nodes:
                split_indices.append(idx)

        # Sort split indices
        split_indices.sort()

        # If no true intersections or only one, keep the whole segment
        if len(split_indices) < 1:
            segment_feature = feature.copy()
            segment_feature['segment_id'] = f"{seg_idx}_0"
            segment_feature['parent_id'] = seg_idx
            segment_feature['is_split'] = False
            # Keep original IDs from parent
            if 'original_ids' in feature:
                segment_feature['original_ids'] = feature['original_ids'].copy()
            else:
                segment_feature['original_ids'] = [feature.get('original_id', str(seg_idx))]
            final_segments.append(segment_feature)
        else:
            if 0 not in split_indices:
                split_indices.insert(0, 0)
            if (len(coords)-1) not in split_indices:
                split_indices.insert(len(coords)-1, len(coords)-1)
            # Split at true intersection nodes
            for seg_idx2 in range(len(split_indices) - 1):
                start_idx = split_indices[seg_idx2]
                end_idx = split_indices[seg_idx2 + 1]

                # Extract segment coordinates
                segment_coords = coords[start_idx:end_idx + 1]

                if len(segment_coords) >= 2:
                    segment_geom = LineString(segment_coords)

                    # Only keep segments longer than tolerance
                    if segment_geom.length > tolerance:
                        segment_feature = feature.copy()
                        segment_feature['geometry_target'] = segment_geom

                        # Extract corresponding source geometry segment
                        source_coords = list(feature['geometry_source'].coords)
                        source_segment_coords = source_coords[start_idx:end_idx + 1]
                        segment_feature['geometry_source'] = LineString(source_segment_coords)

                        segment_feature['segment_id'] = f"{seg_idx}_{seg_idx2}"
                        segment_feature['parent_id'] = seg_idx
                        if segment_feature['parent_id'] == 924166329317558:
                            print(1)
                        segment_feature['is_split'] = True
                        segment_feature['split_start_idx'] = start_idx
                        segment_feature['split_end_idx'] = end_idx
                        # Keep original IDs from parent (for split segments, they inherit the same IDs)
                        if 'original_ids' in feature:
                            segment_feature['original_ids'] = feature['original_ids'].copy()
                        else:
                            segment_feature['original_ids'] = [feature.get('original_id', str(seg_idx))]
                        final_segments.append(segment_feature)

    print(f"  Final segments after splitting: {len(final_segments)}")

    # Step 7: Verify intersection weights for final segments
    print("  Verifying intersection weights for final segments...")

    # Build final node graph for verification
    final_node_graph = defaultdict(lambda: {'connection_weight': 0})

    for segment in final_segments:
        if 'geometry_target' not in segment:
            continue

        geom = segment['geometry_target']
        if geom.geom_type != 'LineString':
            continue

        coords = list(geom.coords)

        for idx, coord in enumerate(coords):
            point = Point(coord)
            key = create_node_key(point)

            is_endpoint = (idx == 0 or idx == len(coords) - 1)

            if is_endpoint:
                final_node_graph[key]['connection_weight'] += 1
            else:
                final_node_graph[key]['connection_weight'] += 2

    # Count nodes with weight >= 3 (true intersections)
    true_intersection_count = sum(1 for node_info in final_node_graph.values()
                                  if node_info['connection_weight'] >= 3)

    # Count segments that connect true intersections
    true_intersection_segments = 0
    for segment in final_segments:
        if segment.get('is_split', False):
            true_intersection_segments += 1

    print(f"  True intersection nodes in final graph: {true_intersection_count}")
    print(f"  Segments between true intersections: {true_intersection_segments}")

    # Step 8: Add connection weight info to each segment's properties
    for segment in final_segments:
        if 'geometry_target' not in segment:
            continue

        geom = segment['geometry_target']
        if geom.geom_type != 'LineString':
            continue

        coords = list(geom.coords)

        # Calculate weights for start and end nodes
        start_point = Point(coords[0])
        end_point = Point(coords[-1])

        start_key = create_node_key(start_point)
        end_key = create_node_key(end_point)

        start_weight = final_node_graph[start_key]['connection_weight']
        end_weight = final_node_graph[end_key]['connection_weight']

        segment['start_node_weight'] = start_weight
        segment['end_node_weight'] = end_weight
        segment['connects_true_intersections'] = (start_weight >= 3 and end_weight >= 3)

    # Add oneway information to each segment
    for segment in final_segments:
        # Get the original oneway status from properties
        props = segment.get('properties', {})
        oneway_value = props.get('oneway') or props.get('oneway:') or props.get('Oneway') or props.get('ONEWAY')

        if oneway_value is None:
            segment['_is_oneway'] = False
            segment['_was_reversed'] = False
        else:
            oneway_str = str(oneway_value).lower().strip()

            if oneway_str in ['no', 'false', '0', 'bidirectional', 'two-way']:
                segment['_is_oneway'] = False
                segment['_was_reversed'] = False
            elif oneway_str in ['yes', 'true', '1', 'forward']:
                segment['_is_oneway'] = True
                segment['_was_reversed'] = False
            elif oneway_str in ['-1', 'reverse', 'backward']:
                segment['_is_oneway'] = True
                segment['_was_reversed'] = True
            else:
                segment['_is_oneway'] = False
                segment['_was_reversed'] = False

    return final_segments


def create_rtree_index(features):
    idx = rtree.index.Index()
    for i, feature in enumerate(features):
        geom = feature['geometry_target']
        bbox = geom.bounds
        idx.insert(i, bbox)

    return idx


def find_potential_matches(features1, features2, idx1, distance_threshold=50):
    """Find potentially close roads using R-tree bounding box intersection."""
    potential_matches = []

    for j, feature2 in enumerate(features2):
        geom2 = feature2['geometry_target']

        # Expand bbox by distance threshold
        minx, miny, maxx, maxy = geom2.bounds
        search_bbox = (
            minx - distance_threshold,
            miny - distance_threshold,
            maxx + distance_threshold,
            maxy + distance_threshold
        )

        # Find intersecting bounding boxes
        intersecting_indices = list(idx1.intersection(search_bbox))

        for i in intersecting_indices:
            geom1 = features1[i]['geometry_target']
            distance = geom1.distance(geom2)

            if distance <= distance_threshold:
                # Calculate additional metrics
                hausdorff_dist = geom1.hausdorff_distance(geom2)
                length1 = geom1.length
                length2 = geom2.length
                length_ratio = min(length1, length2) / max(length1, length2) if max(length1, length2) > 0 else 0

                # Calculate overlap percentage
                buffer1 = geom1.buffer(distance_threshold)
                buffer2 = geom2.buffer(distance_threshold)
                intersection = buffer1.intersection(buffer2)
                union = buffer1.union(buffer2)
                overlap_ratio = intersection.area / union.area if union.area > 0 else 0

                # Calculate directional similarity
                if length1 > 0 and length2 > 0:
                    # Get vectors from start to end
                    vec1 = np.array(geom1.coords[-1]) - np.array(geom1.coords[0])
                    vec2 = np.array(geom2.coords[-1]) - np.array(geom2.coords[0])

                    # Normalize
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)

                    if norm1 > 0 and norm2 > 0:
                        vec1_norm = vec1 / norm1
                        vec2_norm = vec2 / norm2

                        # Cosine similarity (directional similarity)
                        directional_similarity = abs(np.dot(vec1_norm, vec2_norm))
                    else:
                        directional_similarity = 0
                else:
                    directional_similarity = 0

                # Get original IDs for tracking
                original_ids1 = features1[i].get('original_ids', [])
                original_ids2 = features2[j].get('original_ids', [])

                potential_matches.append({
                    'i': i,
                    'j': j,
                    'distance': distance,
                    'hausdorff_distance': hausdorff_dist,
                    'length_ratio': length_ratio,
                    'overlap_ratio': overlap_ratio,
                    'directional_similarity': directional_similarity,
                    'length1': length1,
                    'length2': length2,
                    'is_split1': features1[i].get('is_split', False),
                    'is_split2': features2[j].get('is_split', False),
                    'is_merged1': features1[i].get('is_merged', False),
                    'is_merged2': features2[j].get('is_merged', False),
                    'connects_true_intersections1': features1[i].get('connects_true_intersections', False),
                    'connects_true_intersections2': features2[j].get('connects_true_intersections', False),
                    'start_weight1': features1[i].get('start_node_weight', 0),
                    'end_weight1': features1[i].get('end_node_weight', 0),
                    'start_weight2': features2[j].get('start_node_weight', 0),
                    'end_weight2': features2[j].get('end_node_weight', 0),
                    'original_ids1': original_ids1,
                    'original_ids2': original_ids2
                })

    return potential_matches


def filter_and_rank_matches(matches, min_length_ratio=0.3, min_overlap_ratio=0.3, min_directional_similarity=0.7):
    """
    Filter and rank matches based on multiple criteria.
    """
    if not matches:
        return []

    # Calculate composite score
    for match in matches:
        # Normalize metrics
        distance_score = 1 - min(match['hausdorff_distance'] / 40, 1.0)
        length_ratio_score = match['length_ratio']
        overlap_score = match['overlap_ratio']
        directional_score = (match['directional_similarity'] + 1) / 2

        # Weighted composite score
        composite_score = (
                0.3 * distance_score +
                0.25 * length_ratio_score +
                0.25 * overlap_score +
                0.2 * directional_score
        )

        match['composite_score'] = composite_score
        match['is_good_match'] = (
                match['length_ratio'] >= min_length_ratio and
                match['overlap_ratio'] >= min_overlap_ratio and
                match['directional_similarity'] >= min_directional_similarity
        )

    # Sort by composite score
    matches.sort(key=lambda x: x['composite_score'], reverse=True)

    # Remove duplicates (keep only best match for each segment)
    filtered_matches = []
    used_indices1 = set()
    used_indices2 = set()

    for match in matches:
        if match['i'] not in used_indices1 and match['j'] not in used_indices2:
            if match['is_good_match']:
                filtered_matches.append(match)
                used_indices1.add(match['i'])
                used_indices2.add(match['j'])

    return filtered_matches


def find_projection_matches_for_unmatched(unmatched_features1, features2, idx2, projection_threshold=50):
    """
    Find projection matches for unmatched roads from dataset1 to dataset2.
    For each unmatched road, buffer it by projection_threshold, find candidate roads in dataset2,
    project road points to candidate roads, and match with the candidate having smallest average projection distance.
    """
    print(f"\n  Finding projection matches for {len(unmatched_features1)} unmatched roads...")
    print(f"  Projection threshold: {projection_threshold}m")

    projection_matches = []

    for i, feature1 in enumerate(unmatched_features1):
        geom1 = feature1['geometry_target']

        if geom1.geom_type != 'LineString':
            continue

        # Buffer the road to find candidate roads in dataset2
        buffer_geom = geom1.buffer(projection_threshold)

        # Find candidate roads in dataset2 that intersect with the buffer
        minx, miny, maxx, maxy = buffer_geom.bounds
        search_bbox = (minx, miny, maxx, maxy)

        candidate_indices = list(idx2.intersection(search_bbox))
        if not candidate_indices:
            continue

        best_candidate = None
        best_avg_distance = float('inf')
        best_max_distance = float('inf')
        best_coverage = 0

        for candidate_idx in candidate_indices:
            feature2 = features2[candidate_idx]
            geom2 = feature2['geometry_target']

            if geom2.geom_type != 'LineString':
                continue

            # Sample points along road1 (use vertices plus some interpolated points)
            coords1 = list(geom1.coords)
            sample_points = []

            # Add all vertices
            for coord in coords1:
                sample_points.append(Point(coord))

            # Add interpolated points if road is long
            if geom1.length > 50:
                num_interpolated = max(3, int(geom1.length / 25))
                for k in range(1, num_interpolated):
                    fraction = k / num_interpolated
                    point = geom1.interpolate(fraction, normalized=True)
                    sample_points.append(point)

            # Project each sample point onto candidate road
            projection_distances = []
            valid_projections = 0

            for point in sample_points:
                # Find nearest point on candidate road
                nearest_point = geom2.interpolate(geom2.project(point))
                distance = point.distance(nearest_point)

                # Only consider projections within threshold
                if distance <= projection_threshold:
                    projection_distances.append(distance)

                    # Check if projection is on the road (not beyond endpoints)
                    proj_dist = geom2.project(nearest_point)
                    if 0 <= proj_dist <= geom2.length:
                        valid_projections += 1

            if not projection_distances:
                continue

            avg_distance = np.mean(projection_distances)
            max_distance = np.max(projection_distances)

            # Calculate coverage (percentage of points that project successfully)
            coverage = valid_projections / len(sample_points)

            # Calculate angle similarity
            vec1 = np.array(geom1.coords[-1]) - np.array(geom1.coords[0])
            vec2 = np.array(geom2.coords[-1]) - np.array(geom2.coords[0])

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 > 0 and norm2 > 0:
                vec1_norm = vec1 / norm1
                vec2_norm = vec2 / norm2
                angle_similarity = abs(np.dot(vec1_norm, vec2_norm))
            else:
                angle_similarity = 0

            # We want roads that are parallel (high angle similarity)
            # and have small average projection distance
            if (avg_distance < best_avg_distance and
                    angle_similarity > 0.7 and  # Roads should be roughly parallel
                    coverage > 0.6):  # At least 60% of points should project successfully

                best_avg_distance = avg_distance
                best_max_distance = max_distance
                best_coverage = coverage
                best_candidate = candidate_idx

        if best_candidate is not None:
            feature2 = features2[best_candidate]

            # Calculate additional metrics for the match
            geom2 = feature2['geometry_target']
            length1 = geom1.length
            length2 = geom2.length
            length_ratio = min(length1, length2) / max(length1, length2) if max(length1, length2) > 0 else 0

            # Calculate directional similarity
            if length1 > 0 and length2 > 0:
                vec1 = np.array(geom1.coords[-1]) - np.array(geom1.coords[0])
                vec2 = np.array(geom2.coords[-1]) - np.array(geom2.coords[0])

                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)

                if norm1 > 0 and norm2 > 0:
                    vec1_norm = vec1 / norm1
                    vec2_norm = vec2 / norm2
                    directional_similarity = abs(np.dot(vec1_norm, vec2_norm))
                else:
                    directional_similarity = 0
            else:
                directional_similarity = 0

            projection_matches.append({
                'i': i,  # Index in unmatched_features1
                'j': best_candidate,  # Index in features2
                'avg_projection_distance': best_avg_distance,
                'max_projection_distance': best_max_distance,
                'coverage': best_coverage,
                'length_ratio': length_ratio,
                'directional_similarity': directional_similarity,
                'length1': length1,
                'length2': length2,
                'is_projection_match': True,
                'is_split1': feature1.get('is_split', False),
                'is_split2': feature2.get('is_split', False),
                'is_merged1': feature1.get('is_merged', False),
                'is_merged2': feature2.get('is_merged', False),
                'connects_true_intersections1': feature1.get('connects_true_intersections', False),
                'connects_true_intersections2': feature2.get('connects_true_intersections', False),
                'original_ids1': feature1.get('original_ids', []),
                'original_ids2': feature2.get('original_ids', [])
            })

    print(f"  Found {len(projection_matches)} projection matches")
    print()
    # Filter projection matches to avoid duplicates
    filtered_projection_matches = []
    used_indices1 = set()
    used_indices2 = set()

    # Sort by average projection distance (smallest first)
    projection_matches.sort(key=lambda x: x['avg_projection_distance'])

    for match in projection_matches:
        if match['i'] not in used_indices1:
            # Additional quality checks
            if (match['avg_projection_distance'] <= projection_threshold and
                    match['coverage'] >= 0.6 and
                    match['directional_similarity'] >= 0.7):
                filtered_projection_matches.append(match)
                used_indices1.add(match['i'])
                used_indices2.add(match['j'])

    print(f"  Filtered to {len(filtered_projection_matches)} high-quality projection matches")

    return filtered_projection_matches


def get_oneway_status(feature):
    """
    Extract oneway status from feature properties.
    Returns: (is_oneway, direction_code, needs_reverse)
    - is_oneway: True if road is one-way, False if bidirectional or no oneway tag
    - direction_code: 'forward', 'reverse', or 'bidirectional'
    - needs_reverse: True if geometry needs to be reversed for comparison
    """
    # First check if we have precomputed oneway status

    if '_is_oneway' in feature:
        is_oneway = feature['_is_oneway']
        was_reversed = feature.get('_was_reversed', False)

        if not is_oneway:
            return False, 'bidirectional', False
        else:
            if was_reversed:
                return True, 'reverse', True
            else:
                return True, 'forward', False

    # Fallback to original properties
    props = feature.get('properties', {})

    # Try different possible oneway property names
    oneway_value = props.get('oneway') or props.get('oneway:') or props.get('Oneway') or props.get('ONEWAY')

    is_junction = props.get('junction')


    if is_junction:
        return True, 'forward', False

    if oneway_value is None:
        # Check if it's bidirectional (explicit 'no' or missing)
        return False, 'bidirectional', False

    # Convert to string and lowercase for comparison
    oneway_str = str(oneway_value).lower().strip()

    # Check for bidirectional roads
    if oneway_str in ['no', 'false', '0', 'bidirectional', 'two-way']:
        return False, 'bidirectional', False

    # Check for forward direction (no reversal needed)
    if oneway_str in ['yes', 'true', '1', 'forward']:
        return True, 'forward', False

    # Check for reverse direction (needs reversal)
    if oneway_str in ['-1', 'reverse', 'backward']:
        return True, 'reverse', True

    # Default to bidirectional if unknown value
    return False, 'bidirectional', False


def reverse_geometry(geom):
    """Reverse a LineString geometry."""
    if geom.geom_type != 'LineString':
        return geom

    coords = list(geom.coords)
    reversed_coords = list(reversed(coords))
    return LineString(reversed_coords)


def compare_directions(feature1, feature2):
    """
    Compare directions of two matched roads considering oneway tags.
    Returns: (direction_status, geometry1_modified, geometry2_modified)
    - direction_status: 'same', 'opposite', 'mixed', 'bidirectional_both',
      'bidirectional_to_oneway_same', 'bidirectional_to_oneway_opposite',
      'oneway_to_bidirectional_same', 'oneway_to_bidirectional_opposite', or 'unknown'
    - geometry1_modified: geometry with proper direction for comparison
    - geometry2_modified: geometry with proper direction for comparison
    """
    # Get oneway status for both features
    is_oneway1, dir_code1, needs_reverse1 = get_oneway_status(feature1)
    is_oneway2, dir_code2, needs_reverse2 = get_oneway_status(feature2)

    if feature1['properties']['id'] == '923961902617558':
        print(1)

    # Get geometries
    geom1_source = feature1['geometry_source']
    geom2_source = feature2['geometry_source']

    # Apply reversals based on oneway tags
    geom1_modified = geom1_source
    geom2_modified = geom2_source

    # Calculate directional similarity between modified geometries
    directional_similarity = None
    if (geom1_modified.geom_type == 'LineString' and
            geom2_modified.geom_type == 'LineString'):

        coords1 = list(geom1_modified.coords)
        coords2 = list(geom2_modified.coords)

        if len(coords1) >= 2 and len(coords2) >= 2:
            # Calculate direction vectors
            vec1 = np.array(coords1[-1]) - np.array(coords1[0])
            vec2 = np.array(coords2[-1]) - np.array(coords2[0])

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 > 0 and norm2 > 0:
                vec1_norm = vec1 / norm1
                vec2_norm = vec2 / norm2

                # Cosine similarity (1 = same direction, -1 = opposite)
                directional_similarity = np.dot(vec1_norm, vec2_norm)

    # Determine direction comparison status
    if not is_oneway1 and not is_oneway2:
        # Both are bidirectional
        direction_status = 'bidirectional_both'


    elif is_oneway1 and not is_oneway2:
        # First is one-way, second is bidirectional
        if directional_similarity is not None:
            if directional_similarity > 0.7:
                direction_status = 'oneway_to_bidirectional_same'
            elif directional_similarity < -0.7:
                direction_status = 'oneway_to_bidirectional_opposite'
            else:
                direction_status = 'oneway_to_bidirectional_mixed'
        else:
            direction_status = 'oneway_to_bidirectional'


    elif not is_oneway1 and is_oneway2:
        # First is bidirectional, second is one-way
        if directional_similarity is not None:
            if directional_similarity > 0.7:
                direction_status = 'bidirectional_to_oneway_same'
            elif directional_similarity < -0.7:
                direction_status = 'bidirectional_to_oneway_opposite'
            else:
                direction_status = 'bidirectional_to_oneway_mixed'
        else:
            direction_status = 'bidirectional_to_oneway'


    else:
        # Both are one-way
        if directional_similarity is not None:
            if directional_similarity > 0.7:
                direction_status = 'same_direction'
            elif directional_similarity < -0.7:
                direction_status = 'opposite_direction'
            else:
                direction_status = 'mixed_direction'
        else:
            direction_status = 'unknown_direction'

    return direction_status, geom1_modified, geom2_modified


def analyze_directional_differences(matches, features1_processed, features2_processed, file_name):
    """
    Analyze directional differences between matched roads.
    Creates a separate GeoJSON file with directional differences.
    """
    print(f"\n  Analyzing directional differences for {len(matches)} matches...")

    directional_differences = []

    for match_idx, match in enumerate(matches):
        i = match['i']
        j = match['j']

        feature1 = features1_processed[i]
        if 'junction' in feature1['properties'] and feature1['properties']['junction'] in ('roundabout', 'circular'):
            continue
        feature2 = features2_processed[j]

        # Get oneway status
        is_oneway1, dir_code1, needs_reverse1 = get_oneway_status(feature1)
        is_oneway2, dir_code2, needs_reverse2 = get_oneway_status(feature2)

        # Compare directions
        direction_status, geom1_modified, geom2_modified = compare_directions(feature1, feature2)

        if feature1['properties']['id'] == '923961902617558':
            print(1)

        # Get properties
        props1 = feature1['properties']
        props2 = feature2['properties']

        name1 = props1.get('name', props1.get('NAME', props1.get('road_name', 'Unknown')))
        name2 = props2.get('name', props2.get('NAME', props2.get('road_name', 'Unknown')))

        # Get original IDs
        original_ids1 = feature1.get('original_ids', [])
        original_ids2 = feature2.get('original_ids', [])

        # Create combined ID string
        if feature1.get('is_merged', False) and len(original_ids1) > 1:
            combined_id1 = '_'.join(str(id) for id in original_ids1)
        elif feature1.get('is_split', False):
            # For split features, use the first ID (they all have the same IDs)
            combined_id1 = original_ids1[0] if original_ids1 else 'unknown'
        else:
            combined_id1 = original_ids1[0] if original_ids1 else 'unknown'

        if feature2.get('is_merged', False) and len(original_ids2) > 1:
            combined_id2 = '_'.join(str(id) for id in original_ids2)
        elif feature2.get('is_split', False):
            combined_id2 = original_ids2[0] if original_ids2 else 'unknown'
        else:
            combined_id2 = original_ids2[0] if original_ids2 else 'unknown'

        # Check if this is a directional difference case
        is_directional_difference = False
        difference_type = ''

        oneway = feature1['properties']['oneway'] if 'oneway' in feature1['properties'] else 'yes'
        # Check for all directional difference cases including the new ones
        if direction_status.startswith('oneway_to_bidirectional'):
            is_directional_difference = True
            difference_type = 'oneway_to_bidirectional'
        elif direction_status.startswith('bidirectional_to_oneway'):
            is_directional_difference = True
            difference_type = 'bidirectional_to_oneway'
        elif direction_status == 'opposite_direction':
            is_directional_difference = True
            difference_type = 'opposite_direction'
            if oneway == 'yes':
                oneway = '-1'
            elif oneway == '-1':
                oneway = 'yes'
        elif direction_status == 'mixed_direction':
            is_directional_difference = True
            difference_type = 'mixed_direction'

        # For bidirectional_to_oneway, add direction info to the direction_status
        direction_display_status = direction_status
        if direction_status.startswith('bidirectional_to_oneway'):
            # Extract the direction info
            if '_same' in direction_status:
                direction_display_status = 'same'
                oneway = 'yes'
            elif '_opposite' in direction_status:
                direction_display_status = 'opposite'
                # if oneway == 'yes':
                #     oneway = '-1'
                # elif oneway == '-1':
                #     oneway = 'yes'
                # elif oneway == 'no':
                #     oneway = '-1'
            elif '_mixed' in direction_status:
                direction_display_status = 'mixed'


        if is_directional_difference:
            crs_3857 = pyproj.CRS("EPSG:3857")
            crs_4326 = pyproj.CRS("EPSG:4326")
            transformer1 = pyproj.Transformer.from_crs(crs_4326, crs_3857, always_xy=True).transform
            transformer2 = pyproj.Transformer.from_crs(crs_3857, crs_4326, always_xy=True).transform
            # # Create output feature
            # geom = feature1['geometry_target'].parallel_offset(6, side='right')
            # output_geom = transform(transformer1, geom)
            # # Create feature for directional differences file
            output_geom = geom1_modified if direction_display_status == 'same' else reverse_geometry(geom1_modified)
            output_geom = transform(transformer1, output_geom)
            geom1 = output_geom.parallel_offset(6, side='right')
            geom2 = output_geom.parallel_offset(6, side='left')
            output_geom1 = transform(transformer2, geom1)
            output_geom2 = transform(transformer2, geom2)
            diff_feature = {
                'type': 'Feature',
                'properties': {
                    'oneway': 'yes',
                    'difference_type': difference_type,
                    'direction_status': direction_display_status,
                    'color': 'pink'
                },
                'geometry': mapping(output_geom1)
            }
            directional_differences.append(diff_feature)

            if difference_type == 'oneway_to_bidirectional':
                diff_feature = {
                    'type': 'Feature',
                    'properties': {
                        'oneway': 'yes',
                        'difference_type': difference_type,
                        'direction_status': direction_display_status,
                        'color': 'red'
                    },
                    'geometry': mapping(reverse_geometry(output_geom2))
                }
                directional_differences.append(diff_feature)

            # Log the difference with direction info
            print(f"  Directional difference found (Match {match_idx + 1}):")
            print(f"    Dataset 1: '{name1}' ({'one-way' if is_oneway1 else 'bidirectional'} - {dir_code1})")
            print(f"    Dataset 2: '{name2}' ({'one-way' if is_oneway2 else 'bidirectional'} - {dir_code2})")
            print(
                f"    Dataset 1 ID: {combined_id1} ({'merged' if feature1.get('is_merged') else 'split' if feature1.get('is_split') else 'single'})")
            print(
                f"    Dataset 2 ID: {combined_id2} ({'merged' if feature2.get('is_merged') else 'split' if feature2.get('is_split') else 'single'})")
            print(f"    Difference type: {difference_type}")
            if direction_status.startswith('bidirectional_to_oneway'):
                if '_same' in direction_status:
                    print(f"    Direction: Bidirectional road goes in SAME direction as oneway road")
                elif '_opposite' in direction_status:
                    print(f"    Direction: Bidirectional road goes in OPPOSITE direction to oneway road")
                elif '_mixed' in direction_status:
                    print(f"    Direction: Direction relationship is MIXED/UNCLEAR")
            print()

    # Save directional differences to GeoJSON file
    if directional_differences:
        print(f"  Found {len(directional_differences)} directional differences")
        save_directional_differences(directional_differences, file_name)
    else:
        print(f"  No directional differences found")

    return directional_differences


def save_directional_differences(directional_differences, output_path):
    """
    Save directional differences to a GeoJSON file.
    """
    print(f"  Saving directional differences to {output_path}...")

    geojson = {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {
                'name': 'EPSG:4326'
            }
        },
        'features': directional_differences
    }

    with open(output_path.replace('.geojson', '_dif.geojson'), 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"  Saved {len(directional_differences)} directional differences")


def save_split_geojson(features_split, output_path, unproject):
    """
    Save split roads to a new GeoJSON file.
    """
    print(f"  Saving split roads to {output_path}...")

    features_geojson = []

    for i, feature in enumerate(features_split):
        # Convert geometry back to EPSG:4326
        geom_source = feature['geometry_source']

        # Create properties with segment info
        properties = feature['properties'].copy()
        properties['segment_id'] = feature.get('segment_id', '')
        properties['parent_id'] = feature.get('parent_id', '')
        properties['is_split'] = feature.get('is_split', False)
        properties['is_merged'] = feature.get('is_merged', False)
        properties['connects_true_intersections'] = feature.get('connects_true_intersections', False)
        properties['start_node_weight'] = feature.get('start_node_weight', 0)
        properties['end_node_weight'] = feature.get('end_node_weight', 0)
        properties['length_meters'] = round(feature['geometry_target'].length, 2)

        # Add ID information
        original_ids = feature.get('original_ids', [])
        if feature.get('is_merged', False) and len(original_ids) > 1:
            # For merged features: combine IDs like id1_id2_id3
            combined_id = '_'.join(str(id) for id in original_ids)
            properties['combined_id'] = combined_id
            properties['original_ids'] = original_ids
        elif feature.get('is_split', False):
            # For split features: use the original ID (all segments have same ID)
            if original_ids:
                properties['original_id'] = original_ids[0]
                properties['original_ids'] = original_ids
        else:
            # For single features
            if original_ids:
                properties['original_id'] = original_ids[0]
                properties['original_ids'] = original_ids

        # Add oneway information
        is_oneway, dir_code, needs_reverse = get_oneway_status(feature)
        properties['is_oneway'] = is_oneway
        properties['direction_code'] = dir_code
        properties['needs_reverse'] = needs_reverse

        # Add merge information if available
        if 'merged_from' in feature:
            properties['merged_from_count'] = len(feature['merged_from'])

        # Create GeoJSON feature
        geojson_feature = {
            'type': 'Feature',
            'properties': properties,
            'geometry': mapping(geom_source)
        }

        features_geojson.append(geojson_feature)

    # Create GeoJSON FeatureCollection
    geojson = {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {
                'name': 'EPSG:4326'
            }
        },
        'features': features_geojson
    }

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"  Saved {len(features_geojson)} segments")

    # Print statistics
    merged_count = sum(1 for f in features_split if f.get('is_merged', False))
    split_count = sum(1 for f in features_split if f.get('is_split', False))
    print(f"  Merged segments: {merged_count}")
    print(f"  Split segments: {split_count}")


def compare_roads_with_projection_matching(geojson1_path, geojson2_path,
                                           distance_threshold=50,
                                           projection_threshold=50,
                                           split_tolerance=0.1):
    """
    Main function to compare roads with weighted intersection detection and projection matching.
    """
    print("=" * 70)
    print("ROAD COMPARISON WITH DIRECTIONAL ANALYSIS")
    print("=" * 70)
    print(f"Direct match threshold: {distance_threshold} meters")
    print(f"Projection match threshold: {projection_threshold} meters")
    print(f"Split tolerance: {split_tolerance} meters")
    print("Intersection detection: endpoint=1, internal point=2, total>=3 = true intersection")
    print("CRS: EPSG:4326 → EPSG:3857 (analysis) → EPSG:4326 (output)")
    print("=" * 70)

    # 1. Load and reproject
    print("\n1. Loading and reprojecting GeoJSON files...")
    features1, project1, unproject1 = load_and_reproject_geojson(geojson1_path)
    features2, project2, unproject2 = load_and_reproject_geojson(geojson2_path)

    print(f"   Dataset 1: {len(features1)} roads")
    print(f"   Dataset 2: {len(features2)} roads")

    # 2. Process roads with weighted intersection detection
    print("\n2. Processing roads with weighted intersection detection...")
    print(f"   Dataset 1:")
    features1_processed = build_road_graph_with_weighted_intersections(features1, split_tolerance)
    print(f"   Dataset 2:")
    features2_processed = build_road_graph_with_weighted_intersections(features2, split_tolerance)

    print(f"\n   After processing:")
    print(f"   Dataset 1: {len(features1_processed)} segments")
    print(f"   Dataset 2: {len(features2_processed)} segments")

    # Count statistics
    merged_count1 = sum(1 for f in features1_processed if f.get('is_merged', False))
    merged_count2 = sum(1 for f in features2_processed if f.get('is_merged', False))
    split_count1 = sum(1 for f in features1_processed if f.get('is_split', False))
    split_count2 = sum(1 for f in features2_processed if f.get('is_split', False))
    true_intersection_segments1 = sum(1 for f in features1_processed if f.get('connects_true_intersections', False))
    true_intersection_segments2 = sum(1 for f in features2_processed if f.get('connects_true_intersections', False))

    print(f"   Merged segments: {merged_count1} / {merged_count2}")
    print(f"   Split segments: {split_count1} / {split_count2}")
    print(f"   Segments between true intersections: {true_intersection_segments1} / {true_intersection_segments2}")

    # 3. Save processed GeoJSON files (with oneway info)
    print("\n3. Saving processed GeoJSON files with oneway information...")
    save_split_geojson(features1_processed, 'roads1_weighted_intersections.geojson', unproject1)
    save_split_geojson(features2_processed, 'roads2_weighted_intersections.geojson', unproject2)

    # 4. Build spatial index
    print("\n4. Building spatial index...")
    idx1 = create_rtree_index(features1_processed)
    idx2 = create_rtree_index(features2_processed)

    # 5. Find direct matches
    print(f"\n5. Finding direct matches within {distance_threshold} meters...")
    all_matches = find_potential_matches(features1_processed, features2_processed, idx1, distance_threshold)

    # 6. Filter and rank direct matches
    print("\n6. Filtering and ranking direct matches...")
    filtered_direct_matches = filter_and_rank_matches(
        all_matches,
        min_length_ratio=0.3,
        min_overlap_ratio=0.3,
        min_directional_similarity=0.7
    )

    print(f"   Found {len(all_matches)} potential direct matches")
    print(f"   Filtered to {len(filtered_direct_matches)} high-quality direct matches")

    # 7. Find projection matches for unmatched roads from dataset1
    # Get indices of roads from dataset1 that are already matched
    matched_indices1 = set(match['i'] for match in filtered_direct_matches)

    # Create list of unmatched features from dataset1
    unmatched_features1 = []
    unmatched_indices1 = []

    for i, feature in enumerate(features1_processed):
        if i not in matched_indices1:
            unmatched_features1.append(feature)
            unmatched_indices1.append(i)

    print(f"\n7. Found {len(unmatched_features1)} unmatched roads in dataset1")
    print("   Looking for projection matches...")

    # Find projection matches for unmatched roads
    projection_matches = find_projection_matches_for_unmatched(
        unmatched_features1,
        features2_processed,
        idx2,
        projection_threshold=projection_threshold
    )

    # Map projection match indices back to original indices in features1_processed
    for match in projection_matches:
        original_i = unmatched_indices1[match['i']]
        match['i'] = original_i

    print(f"   Found {len(projection_matches)} projection matches")

    # 8. Combine all matches
    print("\n8. Combining all matches...")
    all_matches_combined = filtered_direct_matches + projection_matches

    # Remove any duplicates (though projection matches should only be for unmatched roads)
    final_matches = []
    used_indices1 = set()
    used_indices2 = set()

    # Sort all matches by type (direct first, then projection) and quality
    all_matches_combined.sort(key=lambda x: (
        0 if not x.get('is_projection_match', False) else 1,  # Direct matches first
        x.get('distance', x.get('avg_projection_distance', float('inf')))  # Then by distance
    ))

    for match in all_matches_combined:
        if match['i'] not in used_indices1:
            # and match['j'] not in used_indices2):
            final_matches.append(match)
            used_indices1.add(match['i'])
            used_indices2.add(match['j'])

    print(f"   Total combined matches: {len(final_matches)}")
    print("-" * 70)

    # 9. Analyze directional differences
    print("\n9. Analyzing directional differences...")
    directional_differences = analyze_directional_differences(
        final_matches,
        features1_processed,
        features2_processed,
        geojson1_path.replace('/1/','/3/')
    )

    # Prepare results
    results = {
        'total_direct_matches': len(all_matches),
        'total_filtered_direct_matches': len(filtered_direct_matches),
        'total_projection_matches': len(projection_matches),
        'total_combined_matches': len(final_matches),
        'directional_differences_count': len(directional_differences),
        'distance_threshold': distance_threshold,
        'projection_threshold': projection_threshold,
        'split_tolerance': split_tolerance,
        'original_counts': {
            'dataset1': len(features1),
            'dataset2': len(features2)
        },
        'processed_counts': {
            'dataset1': len(features1_processed),
            'dataset2': len(features2_processed),
            'merged1': merged_count1,
            'merged2': merged_count2,
            'split1': split_count1,
            'split2': split_count2,
            'true_intersection_segments1': true_intersection_segments1,
            'true_intersection_segments2': true_intersection_segments2
        },
        'matches': []
    }

    output_features = []

    # Display and process matches
    print("\nTop matches:")
    display_count = min(30, len(final_matches))
    for match_idx, match in enumerate(final_matches):
        i = match['i']
        j = match['j']

        feature1 = features1_processed[i]
        feature2 = features2_processed[j]

        props1 = feature1['properties']
        props2 = feature2['properties']

        # Get road names
        name1 = props1.get('name', props1.get('NAME', props1.get('road_name', 'Unknown')))
        name2 = props2.get('name', props2.get('NAME', props2.get('road_name', 'Unknown')))

        match_type = "PROJECTION" if match.get('is_projection_match', False) else "DIRECT"

        print(f"Match {match_idx + 1} ({match_type}):")
        print(f"  Dataset 1: '{name1}' (segment {feature1.get('segment_id', '0')})")
        print(f"  Dataset 2: '{name2}' (segment {feature2.get('segment_id', '0')})")

        # Show ID information
        original_ids1 = feature1.get('original_ids', [])
        original_ids2 = feature2.get('original_ids', [])

        if feature1.get('is_merged', False) and len(original_ids1) > 1:
            combined_id1 = '_'.join(str(id) for id in original_ids1)
            print(f"  Dataset 1 ID (merged): {combined_id1}")
        elif feature1.get('is_split', False):
            print(f"  Dataset 1 ID (split): {original_ids1[0] if original_ids1 else 'unknown'}")
        else:
            print(f"  Dataset 1 ID: {original_ids1[0] if original_ids1 else 'unknown'}")

        if feature2.get('is_merged', False) and len(original_ids2) > 1:
            combined_id2 = '_'.join(str(id) for id in original_ids2)
            print(f"  Dataset 2 ID (merged): {combined_id2}")
        elif feature2.get('is_split', False):
            print(f"  Dataset 2 ID (split): {original_ids2[0] if original_ids2 else 'unknown'}")
        else:
            print(f"  Dataset 2 ID: {original_ids2[0] if original_ids2 else 'unknown'}")

        if match_type == "PROJECTION":
            print(f"  Avg projection distance: {match['avg_projection_distance']:.2f}m")
            print(f"  Max projection distance: {match['max_projection_distance']:.2f}m")
            print(f"  Coverage: {match['coverage']:.2%}")
        else:
            print(f"  Distance: {match['distance']:.2f}m")
            print(f"  Length ratio: {match['length_ratio']:.2f}")
            print(f"  Overlap: {match['overlap_ratio']:.2f}")

        match_feature = {
            'type': 'Feature',
            'properties': {
                'match_id': f"match_{match_idx:03d}",
                'match_type': match_type,
                'distance_m': float(match.get('distance', match.get('avg_projection_distance', 0))),
                'hausdorff_m': float(match.get('hausdorff_distance', match.get('max_projection_distance', 0))),
                'length_ratio': float(match['length_ratio']),
                'overlap_ratio': float(match.get('overlap_ratio', match.get('coverage', 0))),
                'directional_similarity': float(match['directional_similarity']),
                'dataset1_name': name1,
                'dataset2_name': name2,
                'dataset1_segment': feature1.get('segment_id', '0'),
                'dataset2_segment': feature2.get('segment_id', '0'),
                'dataset1_parent': feature1.get('parent_id', ''),
                'dataset2_parent': feature2.get('parent_id', ''),
                'dataset1_length_m': float(match['length1']),
                'dataset2_length_m': float(match['length2']),
                'dataset1_merged': match.get('is_merged1', False),
                'dataset2_merged': match.get('is_merged2', False),
                'dataset1_split': match.get('is_split1', False),
                'dataset2_split': match.get('is_split2', False),
                'connects_true_intersections1': match.get('connects_true_intersections1', False),
                'connects_true_intersections2': match.get('connects_true_intersections2', False)
            },
            'geometry': {
                'type': 'GeometryCollection',
                'geometries': [
                    mapping(feature1['geometry_source']),
                    # mapping(feature2['geometry_source'])
                ]
            }
        }

        output_features.append(match_feature)

        # Store match info
        match_info = {
            'match_id': f"match_{match_idx:03d}",
            'match_type': match_type,
            'dataset1_index': i,
            'dataset2_index': j,
            'dataset1_segment': feature1.get('segment_id', '0'),
            'dataset2_segment': feature2.get('segment_id', '0'),
            'distance': float(match.get('distance', match.get('avg_projection_distance', 0))),
            'hausdorff_distance': float(match.get('hausdorff_distance', match.get('max_projection_distance', 0))),
            'length_ratio': float(match['length_ratio']),
            'overlap_ratio': float(match.get('overlap_ratio', match.get('coverage', 0))),
            'directional_similarity': float(match['directional_similarity']),
            'dataset1_name': name1,
            'dataset2_name': name2,
            'dataset1_length': float(match['length1']),
            'dataset2_length': float(match['length2']),
            'dataset1_merged': match.get('is_merged1', False),
            'dataset2_merged': match.get('is_merged2', False),
            'dataset1_split': match.get('is_split1', False),
            'dataset2_split': match.get('is_split2', False),
            'connects_true_intersections1': match.get('connects_true_intersections1', False),
            'connects_true_intersections2': match.get('connects_true_intersections2', False)
        }

        # Add projection-specific info if applicable
        if match_type == "PROJECTION":
            match_info.update({
                'avg_projection_distance': float(match['avg_projection_distance']),
                'max_projection_distance': float(match['max_projection_distance']),
                'coverage': float(match['coverage'])
            })

        results['matches'].append(match_info)

    if len(final_matches) > display_count:
        print(f"... and {len(final_matches) - display_count} more matches")

    # Create output GeoJSON for matches
    output_geojson = {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {'name': 'EPSG:4326'}
        },
        'features': output_features
    }

    results['output_geojson'] = output_geojson

    return results


def export_results(results, base_output_path='road_comparison_with_projections'):
    json_output_path = f'{base_output_path}_results.json'
    with open(json_output_path, 'w') as f:
        serializable_results = {
            'total_direct_matches': results['total_direct_matches'],
            'total_filtered_direct_matches': results['total_filtered_direct_matches'],
            'total_projection_matches': results['total_projection_matches'],
            'total_combined_matches': results['total_combined_matches'],
            'directional_differences_count': results['directional_differences_count'],
            'distance_threshold_meters': results['distance_threshold'],
            'projection_threshold_meters': results['projection_threshold'],
            'split_tolerance_meters': results['split_tolerance'],
            'original_counts': results['original_counts'],
            'processed_counts': results['processed_counts'],
            'matches': results['matches']
        }
        json.dump(serializable_results, f, indent=2, default=str)

    # Export matched roads to GeoJSON
    geojson_output_path = f'{base_output_path}_matches.geojson'
    with open(geojson_output_path, 'w') as f:
        json.dump(results['output_geojson'], f, indent=2)

    print(f"\n10. Results exported:")
    print(f"   - Processed dataset 1: roads1_weighted_intersections.geojson")
    print(f"   - Processed dataset 2: roads2_weighted_intersections.geojson")
    print(f"   - Directional differences: directional_differences.geojson")
    print(f"   - Detailed results: {json_output_path}")
    print(f"   - Matched roads (GeoJSON): {geojson_output_path}")
    print("=" * 70)

base = "/Users/hlebtkach/Downloads/comparison/"

if __name__ == "__main__":
    o = Path("/Users/hlebtkach/Downloads/comparison/1")
    for file in o.rglob("*.geojson"):
        geojson1_path = base + '1/' + file.name
        #geojson1_path = "/Users/hlebtkach/Downloads/033111310312_mapbox.geojson"
        #geojson2_path = "/Users/hlebtkach/Downloads/033111310312_here.geojson"
        geojson2_path = base  + '2/' + file.name
        # Parameters
        distance_threshold = 20  # meters for direct matching
        projection_threshold = 15  # meters for projection matching
        split_tolerance = 0.1  # meters

        results = compare_roads_with_projection_matching(
            geojson1_path,
            geojson2_path,
            distance_threshold=distance_threshold,
            projection_threshold=projection_threshold,
            split_tolerance=split_tolerance
        )

        export_results(results, '/Users/hlebtkach/Downloads/033111312111_with_directional_analysis')

        print(f"\nAnalysis complete!")
        print(f"Direct matches: {results['total_filtered_direct_matches']}")
        print(f"Projection matches: {results['total_projection_matches']}")
        print(f"Total matches: {results['total_combined_matches']}")
        print(f"Directional differences found: {results['directional_differences_count']}")
        print(
            f"Match rate: {results['total_combined_matches'] / results['processed_counts']['dataset1'] * 100:.1f}% of dataset1 segments matched")
