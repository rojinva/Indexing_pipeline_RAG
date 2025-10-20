import asyncio
import os
import tempfile
import urllib.parse
from typing import Any, List
from datetime import datetime, timezone
from ...base.processor import BaseProcessor
from ...models.request import StorageMetadata
from langchain_community.document_loaders import PyPDFLoader
from ...models.records import FinanceSearchIndexRecord
from ...clients.azure_blob_service_client import AzureBlobServiceClient
from ...models.constants import FileProcessorName
from ...utils.hashing import generate_hash

from langchain_openai import AzureChatOpenAI

from dotenv import load_dotenv
load_dotenv()

llm = AzureChatOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY_USE2"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_USE2"],
    api_version="2024-12-01-preview",
    azure_deployment="gpt-4o-gs",
    model="gpt-4o",
    temperature=0.0,
    streaming=False,
    max_retries=5,
    max_tokens=2000
)

from typing import Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    """Extracting key information from the document."""

    title: str = Field(..., description="The official title of the document, typically not the first line but a key identifier of its content.")
    contributing_authors: List[str] = Field(
        ..., description="A list of all authors who contributed to the creation of the document."
    )
    publication_date: datetime = Field(
        ..., description="The date when the document was published or officially created."
    )

structured_llm = llm.with_structured_output(Metadata)


class FinancePDFProcessor(BaseProcessor):
    """
    A custom processor for processing finance-related PDF documents.
    """
    def __init__(self, processor_name: str = FileProcessorName.FINANCE_PDF_PROCESSOR):
        super().__init__(processor_name)

    @staticmethod
    def _get_company_name(url: str) -> str:
        # Parse the URL and decode the path to handle characters like %20
        parsed_url = urllib.parse.urlparse(url)
        decoded_path = urllib.parse.unquote(parsed_url.path)
        
        # Split the path into segments and filter out empty segments.
        segments = [seg for seg in decoded_path.split('/') if seg]
        
        if not segments:
            return ""

        # For Azure Blob URLs, the first segment is the container name.
        # The remaining segments are the blob path.
        blob_path = segments[1:]
        
        # If the last segment contains a dot, assume it is a file and remove it.
        if blob_path and '.' in blob_path[-1]:
            directories = blob_path[:-1]
        else:
            directories = blob_path

        # We are after the second subdirectory (i.e., inside the blob_path,
        # assume the first directory is the parent; then the first subdirectory is at index 1
        # and the second subdirectory is at index 2).
        if len(directories) >= 3:
            return directories[2]
        else:
            return ""

    def process(self, storage_metadata: StorageMetadata, text_splitter: Any, datasource_client: AzureBlobServiceClient) -> List[FinanceSearchIndexRecord]:
        """
        Process a finance-related PDF document by downloading it from Azure Blob Storage,
        splitting its content into chunks, extracting key metadata from the first chunk,
        and then combining all the information into a list of FinanceSearchIndexRecord objects.

        The following mappings are applied:
          - row: page number from chunk metadata [chunk.metadata.get("page") + 1]
          - sheet_name: set to an empty string.
          - chunk_num: the index when iterating over the chunked documents.
          - title, contributing_authors, publication_date: extracted from the first chunk.
        """
        file_buffer = None  # Initialize file_buffer for cleanup in finally block
        temp_file_path = None
        try:
            blob_uri = storage_metadata.blob_uri
            # Step 1: Download the blob content.
            file_buffer = datasource_client.download_blob(blob_uri)

            # Step 2: Write the file buffer to a temporary file.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_buffer.read())
                temp_file_path = temp_file.name

            # Step 3: Load the PDF using PyPDFLoader.
            loader = PyPDFLoader(temp_file_path)

            # Step 4: Load and split the document into chunks using the provided text splitter.
            chunked_documents = loader.load_and_split(text_splitter=text_splitter)

            # Step 5: Extract metadata (title, contributing_authors, publication_date) from the first chunk.
            info = structured_llm.invoke(chunked_documents[0].page_content)
            title = info.title
            contributing_authors = ', '.join(info.contributing_authors)
            publication_date = info.publication_date.astimezone(timezone.utc).isoformat()

            results = []
            for idx, chunk in enumerate(chunked_documents):
                # Get the page number from chunk metadata; default to idx+1 if not available.
                row_num = chunk.metadata.get("page") + 1

                # Create a hash value for integrity verification.
                chunk_hash = generate_hash(chunk.page_content)
                
                record = FinanceSearchIndexRecord(
                    parent_path=blob_uri,  # As provided in StorageMetadata
                    use_case="Finance Document",  # You may customize this as needed
                    content=chunk.page_content,
                    chunk_num=idx,
                    chunk_hash=chunk_hash,
                    row=row_num,
                    sheet_name="",
                    title=title,
                    contributing_authors=contributing_authors,
                    publication_date=publication_date,
                    company_name=self._get_company_name(blob_uri)
                )
                results.append(record)
            
            return results

        except Exception as e:
            # Log error if needed and raise a descriptive error.
            raise RuntimeError(f"Failed to process the PDF document: {str(e)}")

        finally:
            # Clean up file buffer if it exists.
            if file_buffer is not None:
                del file_buffer
            # Delete the temporary file if it was created.
            if temp_file_path is not None:
                try:
                    os.remove(temp_file_path)
                except Exception as cleanup_error:
                    print(f"Failed to delete temporary file: {cleanup_error}")

    async def aprocess(self, storage_metadata: StorageMetadata, text_splitter: Any, datasource_client: AzureBlobServiceClient) -> List[FinanceSearchIndexRecord]:
        """
        Asynchronous version of the process method.
        """
        # Ensure that the process method is called asynchronously
        return await asyncio.to_thread(self.process, storage_metadata, text_splitter, datasource_client)
