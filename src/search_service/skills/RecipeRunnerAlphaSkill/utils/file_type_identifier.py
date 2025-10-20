from typing import Optional
from urllib.parse import urlparse
from ..models.constants import FileType

class FileTypeIdentifier:
    @staticmethod
    def identify_file_type(blob_uri: str) -> Optional[FileType]:
        # Parse the URI to extract the file extension
        parsed_uri = urlparse(blob_uri)
        path = parsed_uri.path
        if not path:
            return None
        
        # Extract the file extension
        file_extension = path.split('.')[-1].lower()
        
        # Match the file extension to the FileType enum
        try:
            return FileType(file_extension)
        except ValueError:
            return None