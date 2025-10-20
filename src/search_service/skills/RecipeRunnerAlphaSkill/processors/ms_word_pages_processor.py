import os
import asyncio
import tempfile
import logging
from typing import List, Optional
from docx import Document
from ..models.records import BaseSearchIndexRecord
from ..utils.hashing import generate_hash
from ..utils.string_parsing import parse_blob_uri_root_directory_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class MSWordProcessor:
    def __init__(self):
        """
        Initializes the processor.
        """

    def _extract_text_from_docx(self, docx_path: str) -> str:
        """
        Extracts all text from a .docx file as a single string using python-docx.
        :param docx_path: Path to the .docx file.
        :return: Combined text from the document.
        """
        document = Document(docx_path)
        combined_text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
        return combined_text

    def process(
        self,
        storage_metadata,
        datasource_client,
        text_splitter=None,
        output_path: Optional[str] = None,
    ) -> List[BaseSearchIndexRecord]:
        """
        Processes .docx files, combines all text into a single string, and chunks it.
        Returns a list of BaseSearchIndexRecord objects.
        """
        # Download the blob content using the datasource_client
        file_buffer = datasource_client.download_blob(storage_metadata.blob_uri)
        temp_file_path = None
        try:
            # Save the downloaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                temp_file.write(file_buffer.read())
                temp_file_path = temp_file.name

            # Extract text from .docx file
            logger.info("Extracting text from .docx file.")
            combined_text = self._extract_text_from_docx(temp_file_path)

            # Chunk the combined text
            logger.info("Chunking the combined text.")
            chunked_contents = [combined_text]
            if text_splitter:
                logger.info("Splitting combined text into chunks using the provided text splitter.")
                chunked_contents = text_splitter.split_text(combined_text)

            # Create BaseSearchIndexRecord objects
            records = []
            for chunk_index, content_chunk in enumerate(chunked_contents, start=1):
                record = BaseSearchIndexRecord(
                    parent_path=storage_metadata.blob_uri,
                    use_case="MSWordProcessor",
                    content=content_chunk,
                    chunk_num=chunk_index,
                    chunk_hash=generate_hash(content_chunk),
                    row=-1,  # Chunk number
                    sheet_name=parse_blob_uri_root_directory_name(
                        storage_metadata.blob_uri
                    ),
                )
                logger.info(f"Created record for chunk {chunk_index}.")
                records.append(record)

            if output_path:
                logger.info(f"Writing output to {output_path}...")
                with open(output_path, "w", encoding="utf-8") as f:
                    for record in records:
                        f.write(f"<!-- Chunk {record.chunk_num} -->\n")
                        f.write(record.content)
                        f.write("\n\n---\n\n")
            return records
        finally:
            # Clean up temporary files
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.error(f"Failed to delete temporary file: {e}")

    async def aprocess(
        self,
        storage_metadata,
        datasource_client,
        text_splitter=None,
        output_path: Optional[str] = None,
    ) -> List[BaseSearchIndexRecord]:
        """
        Asynchronously processes .docx files, combines all text into a single string, and chunks it.
        Returns a list of BaseSearchIndexRecord objects.
        """
        # Download the blob content using the datasource_client
        file_buffer = datasource_client.download_blob(storage_metadata.blob_uri)
        temp_file_path = None
        try:
            # Save the downloaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                temp_file.write(file_buffer.read())
                temp_file_path = temp_file.name

            # Extract text from .docx file
            logger.info("Extracting text from .docx file.")
            combined_text = self._extract_text_from_docx(temp_file_path)

            # Chunk the combined text
            logger.info("Chunking the combined text.")
            chunked_contents = [combined_text]
            if text_splitter:
                logger.info("Splitting combined text into chunks using the provided text splitter.")
                chunked_contents = text_splitter.split_text(combined_text)

            async def process_chunk(chunk_index: int, content_chunk: str) -> BaseSearchIndexRecord:
                record = BaseSearchIndexRecord(
                    parent_path=storage_metadata.blob_uri,
                    use_case="MSWordProcessor",
                    content=content_chunk,
                    chunk_num=chunk_index,
                    chunk_hash=generate_hash(content_chunk),
                    row=-1,  # Chunk number
                    sheet_name=parse_blob_uri_root_directory_name(
                        storage_metadata.blob_uri
                    ),
                )
                logger.info(f"Created record for chunk {chunk_index}.")
                return record

            # Process chunks concurrently
            tasks = [
                process_chunk(chunk_index, content_chunk)
                for chunk_index, content_chunk in enumerate(chunked_contents, start=1)
            ]
            records = await asyncio.gather(*tasks)

            if output_path:
                logger.info(f"Writing output to {output_path}...")
                with open(output_path, "w", encoding="utf-8") as f:
                    for record in records:
                        f.write(f"<!-- Chunk {record.chunk_num} -->\n")
                        f.write(record.content)
                        f.write("\n\n---\n\n")
            return records
        finally:
            # Clean up temporary files
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.error(f"Failed to delete temporary file: {e}")