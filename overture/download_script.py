import json
import pandas as pd
from geojson import Feature, FeatureCollection
from overturemaps import core

bbox = (-1.251068,37.714245,-1.050568,38.152917)

core.ALL_RELEASES = ['2025-11-19.0']

gdf = core.geodataframe("segment", bbox=bbox)

df = gdf.assign(geometry=gdf["geometry"].apply(lambda p: p.wkt))

# Export to CSV
df.to_csv("/Users/hlebtkach/Downloads/overture.csv", index=False)

features = []

highways = ['motorway', 'trunk', 'motorway_link', 'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'unclassified', 'residential', 'tertiary_link', 'living_street', 'service']

for _, row in gdf.iterrows():
    try:
        has_osm = False
        if 'sources' in row and row['sources'] is not None:
            sources = row['sources']
            if sources[0]['dataset'].lower() != 'openstreetmap' and sources[0]['dataset'].lower() != 'osm':
                if row['subtype'] == 'road':
                    if row['class'] in highways:
                        feature = Feature(geometry=row['geometry'], properties={})
                        features.append(feature)

    except Exception as e:
        print(f"Error processing row: {e}")
        print(f"Problematic sources value: {row.get('sources', 'No sources field')}")
        continue

# Create and save FeatureCollection
if features:
    feature_collection = FeatureCollection(features)

    with open("/Users/hlebtkach/Downloads/overture_murcia_2.geojson", 'w') as f:
        json.dump(feature_collection, f, indent=2)

    print(f"Successfully saved {len(features)} non-OSM segments to boston_segments_non_osm.geojson")
else:
    print("No non-OSM segments found")