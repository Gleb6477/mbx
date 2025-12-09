import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from shapely import line_merge, MultiLineString, LineString
from shapely.ops import unary_union

import requests
from requests.adapters import HTTPAdapter, Retry
from requests.exceptions import HTTPError
from datetime import datetime
from requests import Response, Session

from toFix.ToFixClient import ToFixClient

tofix_host = ''

tofix_token_name = ''

tofix_token = ''


with ToFixClient(tofix_host,  tofix_token) as tofix_client:
    tofix_project = tofix_client.find_project_by_name("rapid_mr_v2")
    if tofix_project is None:
        tofix_client.create_project("rapid_mr_v2")
        print("created")
    else:
        pr_id = tofix_project["id"]
        print(pr_id)
        print("already existed")
        with open("/Users/hlebtkach/Downloads/133002110.geojson", 'r+',
                  encoding='utf-8') as w:
            data = json.load(w)
            for feature in data['features']:
                feature_id = feature['properties']['id']
                coord = []
                if feature["geometry"]["type"] == 'LineString':
                    coord = feature["geometry"]["coordinates"][0]
                elif feature["geometry"]["type"] == 'MultiLineString':
                    coord = feature["geometry"]["coordinates"][0][0]
                feature["properties"] = {f'string:{k}': v for k, v in feature["properties"].items()}
                feature["properties"]["tofix:category"] = "Missing road"
                print(feature)
                collection = {
                    "type": "FeatureCollection",
                    "features": [feature]
                }
                date = str(datetime.now().date())
                tofix_client.create_or_update_item(pr_id, feature_id, "instruction", coord, collection, dt=date)
