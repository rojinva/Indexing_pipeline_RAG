from urllib.parse import unquote, urlparse

def parse_blob_uri_root_directory_name(blob_uri: str) -> str:
    """
    Parse the blob_uri to extract the root directory name (after the container name).
    For an Azure Blob URL of the form:
        https://<account>.blob.core.windows.net/<container>/<directory>/...
    this method returns <directory> (after URL-decoding)
    """
    parsed = urlparse(blob_uri)
    path_components = parsed.path.split("/")
    # path_components[0] will be empty because the path starts with '/'
    # path_components[1] is the container name and path_components[2] should be the parent directory.
    if len(path_components) >= 3:
        return unquote(path_components[2])
    return ""