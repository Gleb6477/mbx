import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from shapely import line_merge, MultiLineString, LineString
from shapely.ops import unary_union

"""To-fix API client."""

from typing import Optional, Iterable, Dict

import requests
from requests.adapters import HTTPAdapter, Retry
from requests.exceptions import HTTPError
from typing import Dict, Iterable, Optional, Union, Callable
from datetime import datetime, date
from requests import Response, Session

HTTP_ERROR_MESSAGE_TEMPLATE = """
Request failed!
Response Status Code: {status_code}
Response Headers: {headers}
Response Content:
{content}
"""

ITEM_EXISTS_MESSAGE = "Item with this id already exists"


class ToFixClient:
    """To-fix API client."""

    def __init__(
            self,
            host: str,
            token: str,
            pool_connections: int = 8,
            pool_maxsize: int = 8,
            max_retries: int = 5,
            backoff_factor: float = 5,
    ) -> None:
        """Initialize a new ToFixClient instance."""
        self._host = host
        self._headers = {"Content-Type": "application/json", "Authorization": "Token {}".format(token),
                         "x-api-key": "*"}
        self._pool_connections = pool_connections
        self._pool_maxsize = pool_maxsize
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor

    def __enter__(self) -> "ToFixClient":
        """Initialize a new Requests session."""
        self._session = requests.Session()
        retry = Retry(total=self._max_retries, backoff_factor=self._backoff_factor,
                      status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(pool_connections=self._pool_connections, pool_maxsize=self._pool_maxsize,
                              max_retries=retry)
        self._session.mount("https://", adapter)
        return self

    def __exit__(self, *args: list) -> None:
        """Close all adapters and as such the session."""
        self._session.close()

    def create_project(self, name: str) -> dict:
        """Create a project."""
        url = "https://{}/v1/projects".format(self._host)
        payload = {"name": name}
        response = self._session.post(url, headers=self._headers, json=payload)
        self._check_response(response)
        return response.json()

    def find_project_by_name(self, name: str, status: str = "all") -> Optional[dict]:
        """Find a project by name."""
        url = "https://{}/v1/projects".format(self._host)
        params = {"name": name, "status": status}
        response = self._session.get(url, headers=self._headers, params=params)
        self._check_response(response)
        projects = response.json()
        return projects[0] if projects else None

    def create_item(self, project_id: str, item_id: str, instructions: str, pin: list, feature_collection: dict,
                    dt: str) -> None:
        """Create an item in a project."""
        url = "https://{}/v1/projects/{}/items".format(self._host, project_id)
        payload = {"id": item_id, "instructions": instructions, "pin": pin, "featureCollection": feature_collection,
                   "submittedAt": dt}
        #print(payload)
        response = self._session.post(url, headers=self._headers, json=payload)
        self._check_response(response)
        return response.json()

    def update_item(self, project_id: str, item_id: str, instructions: str, pin: list,
                    feature_collection: dict) -> dict:
        """Update a project item."""
        url = "https://{}/v1/projects/{}/items/{}".format(self._host, project_id, item_id)
        payload = {"instructions": instructions, "pin": pin, "featureCollection": feature_collection}
        response = self._session.put(url, headers=self._headers, json=payload)
        self._check_response(response)
        return response.json()

    def update_item_status(self, project_id: str, item_id: str, status: str) -> None:
        """Update the status of an item."""
        url = f"https://{self._host}/v1/projects/{project_id}/items/{item_id}"
        self._check_response(self._session.put(url, headers=self._headers, json={"lock": "locked"}))
        self._check_response(self._session.put(url, headers=self._headers, json={"status": status}))
        self._check_response(self._session.put(url, headers=self._headers, json={"lock": "unlocked"}))

    def update_reviewed_at(self, project_id: str, item_id: str) -> None:
        """Update the reviewedAt field for an item."""
        item = self.get_item(project_id, item_id)
        assert item is not None, f"Failed to find item {item_id} in project {project_id}"
        url = f"https://{self._host}/v1/projects/{project_id}/items/{item_id}"
        self._check_response(self._session.put(url, headers=self._headers, json={"lock": "locked"}))
        self._check_response(self._session.put(url, headers=self._headers, json={"status": "open"}))
        self._check_response(self._session.put(url, headers=self._headers, json={"status": item["status"]}))
        self._check_response(self._session.put(url, headers=self._headers, json={"lock": "unlocked"}))

    def update_item_with_status(
            self, project_id: str, item_id: str, instructions: str, pin: list, feature_collection: dict, status: str,
            dt: datetime
    ) -> dict:
        """Update a project item."""
        url = "https://{}/v1/projects/{}/items/{}".format(self._host, project_id, item_id)
        response = self._session.put(url, headers=self._headers, json={"lock": "locked"})
        self._check_response(response)
        response = self._session.put(
            url,
            headers=self._headers,
            json={
                "status": status,
                "instructions": instructions,
                "pin": pin,
                "featureCollection": feature_collection,
                "submittedAt": dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            },
        )
        self._check_response(response)
        response = self._session.put(url, headers=self._headers, json={"lock": "unlocked"})
        self._check_response(response)
        return response.json()

    def create_or_update_item(self, project_id: str, item_id: str, instructions: str, pin: list,
                              feature_collection: dict, dt: str) -> dict:
        """Create or update an item in a project."""
        url = "https://{}/v1/projects/{}/items".format(self._host, project_id)
        payload = {"instructions": instructions, "pin": pin, "featureCollection": feature_collection}
        response = self._session.post(url, headers=self._headers, json={"id": item_id, **payload, "submittedAt": dt})
        if response.status_code == requests.codes.bad_request and response.json()["message"] == ITEM_EXISTS_MESSAGE:
            response = self._session.put("{}/{}".format(url, item_id), headers=self._headers, json=payload)
            print(f"error processing id={item_id}")
        #self._check_response(response)
        return response.json()

    def reopen_item(self, project_id: str, item_id: str, instructions: str, pin: list, feature_collection: dict,
                    dt: datetime) -> dict:
        """Create or update an item in a project."""
        url = "https://{}/v1/projects/{}/items/{}".format(self._host, project_id, item_id)
        response = self._session.put(url, headers=self._headers, json={"lock": "locked"})
        self._check_response(response)
        response = self._session.put(
            url,
            headers=self._headers,
            json={
                "status": "open",
                "instructions": instructions,
                "pin": pin,
                "featureCollection": feature_collection,
                "submittedAt": dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            },
        )
        self._check_response(response)
        response = self._session.put(url, headers=self._headers, json={"lock": "unlocked"})
        self._check_response(response)
        return response.json()

    def get_items(
            self,
            project_id: str,
            page_size: int = 1000,
            reviewed_after: Optional[Union[date, datetime]] = None,
            statuses: Optional[Iterable[str]] = None,
            fields: Optional[Iterable[str]] = None,
    ) -> list:
        """Get a list of items for a project."""
        url = "https://{}/v1/projects/{}/items".format(self._host, project_id)
        all_items = []
        page = 0
        while page is not None:
            params: Dict[str, Union[int, str]] = {"page": page, "page_size": page_size}
            if reviewed_after is not None:
                params["reviewed_after"] = reviewed_after.isoformat()
            if statuses is not None:
                params["status"] = ",".join(statuses)
            if fields is not None:
                params["fields"] = ",".join(fields)
            response = self._session.get(url, headers=self._headers, params=params)
            self._check_response(response)
            items = response.json()
            all_items.extend(items)
            page = page + 1 if len(items) == page_size else None  # type: ignore
        return all_items

    def get_item(self, project_id: str, item_id: str, fields: Optional[Iterable[str]] = None) -> Optional[dict]:
        """Get an item in a project."""
        url = "https://{}/v1/projects/{}/items/{}".format(self._host, project_id, item_id)
        params: Dict[str, Union[int, str]] = {}
        if fields is not None:
            params["fields"] = ",".join(fields)
        response = self._session.get(url, headers=self._headers, params=params)
        if response.status_code == requests.codes.not_found:
            return None
        self._check_response(response)
        return response.json()

    def get_items_page(self, project_id: str, page: int, page_size: int = 1000,
                       filter_params: Dict[str, str] = None) -> list:
        """Get a list of items for a project page."""
        params_part = "&".join(map(lambda i: "=".join(i), filter_params.items())) if filter_params else ""
        url = "https://{}/v1/projects/{}/items?{}".format(self._host, project_id, params_part)
        all_items = []
        params: dict = {"page": page, "page_size": page_size}
        response = self._session.get(url, headers=self._headers, params=params)
        self._check_response(response)
        items = response.json()
        all_items.extend(items)
        return all_items

    def get_item_tags(self, project_id: str, item_id: str) -> list:
        """Get all tags for an item."""
        url = "https://{}/v1/projects/{}/items/{}/tags".format(self._host, project_id, item_id)
        response = self._session.get(url, headers=self._headers)
        self._check_response(response)
        return response.json()

    def create_item_tag(self, project_id: str, item_id: str, tag_id: str) -> dict:
        """Get all tags for an item."""
        url = "https://{}/v1/projects/{}/items/{}/tags".format(self._host, project_id, item_id)
        response = self._session.post(url, headers=self._headers, json={"tag": tag_id})
        self._check_response(response)
        return response.json()

    def get_project_tags(self, project_id: str) -> list:
        """Get all project tags."""
        url = "https://{}/v1/projects/{}/tags".format(self._host, project_id)
        response = self._session.get(url, headers=self._headers)
        self._check_response(response)
        return response.json()

    def create_project_tag(self, project_id: str, tag_name: str) -> dict:
        """Create project tag."""
        url = "https://{}/v1/projects/{}/tags".format(self._host, project_id)
        response = self._session.post(url, headers=self._headers, json={"name": tag_name})
        self._check_response(response)
        return response.json()

    def get_item_comments(self, project_id: str, item_id: str) -> list:
        """Get a list of comments for an item."""
        url = "https://{}/v1/projects/{}/items/{}/comments".format(self._host, project_id, item_id)
        response = self._session.get(url, headers=self._headers)
        self._check_response(response)
        return response.json()

    def create_item_comment(self, project_id: str, item_id: str, comment_body: str) -> dict:
        """Create a comment for an item."""
        url = "https://{}/v1/projects/{}/items/{}/comments".format(self._host, project_id, item_id)
        response = self._session.post(url, headers=self._headers, json={"body": comment_body})
        self._check_response(response)
        return response.json()

    def get_tags(self, project_id: str) -> list:
        """Get all project tags."""
        url = f"https://{self._host}/v1/projects/{project_id}/tags"
        resource = self._session.get(url, headers=self._headers)
        self._check_response(resource)
        return resource.json()

    @staticmethod
    def _check_response(response: Response) -> None:
        if response.status_code != requests.codes.ok:
            raise HTTPError(
                HTTP_ERROR_MESSAGE_TEMPLATE.format(status_code=response.status_code, headers=response.headers,
                                                   content=response.text))

    def do_with_lock(self, project_id: str, item_id: str, action: Callable[[Session], dict]) -> dict:
        """Lock item, execute action, unlock item."""
        url_entity = "https://{}/v1/projects/{}/items/{}".format(self._host, project_id, item_id)
        response = self._session.put(url_entity, headers=self._headers, json={"lock": "locked"})
        self._check_response(response)
        action_response = action(self._session)
        self._check_response(response)
        response = self._session.put(url_entity, headers=self._headers, json={"lock": "unlocked"})
        self._check_response(response)
        return action_response


def get_tofix_token(token_name: str) -> str:
    """Get tofix token."""
    client = boto3.client("secretsmanager", region_name="us-east-1")
    try:
        response = client.get_secret_value(SecretId=token_name)
        return response["SecretString"]
    except ClientError as e:
        print(f"Unable to get token from {token_name}: {e}")
        raise
