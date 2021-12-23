import argparse
import json
import os

from google.auth.transport import requests
from googleapiclient.errors import HttpError
from google.oauth2 import service_account

def get_session(service_account_json):
    """Returns an authorized Requests Session class using the service account
    credentials JSON. This class is used to perform requests to the
    Healthcare API endpoint."""

    # Pass in the credentials and project ID. If none supplied, get them
    # from the environment.
    credentials = service_account.Credentials.from_service_account_file(
        service_account_json)
    scoped_credentials = credentials.with_scopes(
        ['https://www.googleapis.com/auth/cloud-platform'])

    # Create a requests Session object with the credentials.
    session = requests.AuthorizedSession(scoped_credentials)

    return session

def get_resource(
        resource_id="d70d8f3e-990a-4bc0-b11f-c87349f5d4eb",
        service_account_json="stephen.lu.2002@gmail.com",
        base_url="https://healthcare.googleapis.com/v1beta1",
        project_id="kaggle-siim-healthcare",
        cloud_region="us-central1",
        dataset_id="siim-pneumothorax",
        fhir_store_id="fhir-masks-train"):
    """Gets a FHIR resource."""
    url = '{}/projects/{}/locations/{}'.format(base_url,
                                               project_id, cloud_region)

    resource_path = '{}/datasets/{}/fhirStores/{}/fhir/DocumentReference/{}'.format(
        url, dataset_id, fhir_store_id, resource_id)

    # Make an authenticated API request
    session = get_session(service_account_json)

    headers = {
        'Content-Type': 'application/fhir+json;charset=utf-8'
    }

    response = session.get(resource_path, headers=headers)
    response.raise_for_status()

    resource = response.json()

    print(json.dumps(resource, indent=2))

    return resource

print(get_resource())


