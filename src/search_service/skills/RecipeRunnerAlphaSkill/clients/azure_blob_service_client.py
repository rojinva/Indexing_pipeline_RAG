import io
from urllib.parse import unquote
from azure.storage.blob import BlobServiceClient
from azure.identity import ClientSecretCredential
from ..models.datasource_specs.adls import ADLSDataSourceSpec
from typing import List


class AzureBlobServiceClient:
    def __init__(self, spec: ADLSDataSourceSpec):
        # Extract credentials from the auth spec.
        tenant_id, client_id, client_secret = spec.auth_secret.get_credentials()

        # Construct the account URL.
        self.account_url = f"https://{spec.storage_account_name}.blob.core.windows.net"

        # Configure Azure credential and instantiate the BlobServiceClient.
        self.credential = ClientSecretCredential(
            tenant_id=tenant_id, client_id=client_id, client_secret=client_secret
        )
        self.blob_service_client = BlobServiceClient(
            account_url=self.account_url, credential=self.credential
        )
        self.container_name = spec.container_name
        self.container_client = self.blob_service_client.get_container_client(
            self.container_name
        )

    @classmethod
    def from_spec(cls, spec: ADLSDataSourceSpec) -> "AzureBlobServiceClient":
        """
        Factory method to create a new AzureBlobServiceClient from a given ADLSDataSourceSpec.
        This allows you to easily create clients from any spec without being limited to a singleton.
        """
        return cls(spec)

    def upload_blob(self, data: bytes, blob_name: str, metadata: dict = None) -> None:
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_client.upload_blob(data, metadata=metadata)

    def download_blob(self, blob_uri: str) -> io.BytesIO:
        """
        Download a blob based on its full URI.

        Args:
            blob_uri (str): The URI of the blob to download.

        Returns:
            io.BytesIO: The bytes content of the blob.
        """
        # Clean the blob URI
        unquoted_blob_uri = unquote(blob_uri)

        # Assuming your blob URI follows the format:
        # https://<account_name>.blob.core.windows.net/<container_name>/<blob_path>
        # This splits out the blob's path within the container.
        try:
            blob_path = unquoted_blob_uri.split(f"/{self.container_name}/", 1)[1]
        except IndexError:
            raise ValueError(
                "Blob URI does not match the expected format. Ensure the container name is in the URI."
            )

        # Create a blob client based on the extracted blob path.
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, blob=blob_path
        )
        # Download the blob's content.
        blob_data = blob_client.download_blob().readall()

        # Optionally, to work with a file-like object:
        file_buffer = io.BytesIO(blob_data)
        # You can return file_buffer if you prefer a file-like object.
        return file_buffer

    def get_blob_metadata_from_uri(self, blob_uri: str) -> dict:
        """
        Retrieve metadata for a specific blob using its URI.

        Args:
            blob_uri (str): The URI of the blob.

        Returns:
            dict: The metadata of the blob.
        """
        unquoted_blob_uri = unquote(blob_uri)
        try:
            blob_path = unquoted_blob_uri.split(f"/{self.container_name}/", 1)[1]
        except IndexError:
            raise ValueError(
                "Blob URI does not match the expected format."
            )
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, blob=blob_path
        )
        metadata = blob_client.get_blob_properties().metadata        
        return metadata

    def list_blobs(self) -> List[str]:
        return [blob.name for blob in self.container_client.list_blobs()]

    def list_filtered_blobs(self, prefix: str) -> List[str]:
        return list(
            self.container_client.list_blobs(
                name_starts_with=prefix, include="metadata"
            )
        )

    def delete_blob(self, blob_name: str) -> None:
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_client.delete_blob()
