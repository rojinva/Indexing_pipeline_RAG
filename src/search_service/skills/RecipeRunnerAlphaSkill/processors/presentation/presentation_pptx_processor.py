import asyncio
import base64
import os
import logging
import tempfile
from datetime import datetime
from typing import Any, List, Optional
from urllib.parse import unquote

from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_community.callbacks.manager import get_openai_callback

from .constants import FileExtensions
from .prompts import (
    INTRODUCTION_SLIDE_PROMPT,
    SUMMARY_PROMPT,
    KEYINSIGHTS_PROMPT,
    TEXTEXTRACTION_PROMPT,
)
from .utils import (
    convert_pptx_to_base64_image_data,
    extract_information_from_base64_image,
    generate_hash,
    generate_base_media_path,
    install_libreoffice_dependencies,
)
from ...base.processor import BaseProcessor
from ...clients.azure_blob_service_client import AzureBlobServiceClient
from ...models.constants import FileProcessorName, SourceModality
from ...models.records import MultimodalSearchIndexRecord
from ...models.request import StorageMetadata
from ...utils.string_parsing import parse_blob_uri_root_directory_name
from ...utils.dependencies import install_poppler_utils

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PresentationPPTXProcessor(BaseProcessor):
    """
    A custom processor for processing presentation PPTX documents.
    """

    def __init__(
        self,
        processor_name: Optional[str] = FileProcessorName.PRESENTATION_PPTX_PROCESSOR,
        rate_limit: Optional[int] = 5,
        upload_image: Optional[bool] = False,
    ):
        super().__init__(processor_name)
        self._semaphore = asyncio.Semaphore(rate_limit)
        self._upload_image = upload_image
        self._total_cost = 0.0
        self._total_tokens = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def get_cost_estimate(self):
        return {
            "total_cost": self._total_cost,
            "total_tokens": self._total_tokens,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
        }

    def process(
        self,
        storage_metadata: StorageMetadata,
        text_splitter: Any,
        datasource_client: AzureBlobServiceClient,
    ) -> List[MultimodalSearchIndexRecord]:
        pass

    async def aprocess(
        self,
        storage_metadata: StorageMetadata,
        text_splitter: Any,
        datasource_client: AzureBlobServiceClient,
    ) -> List[MultimodalSearchIndexRecord]:
        """
        Asynchronously process a PowerPoint presentation document for multimodal search indexing.

        This method downloads a PPTX file from Azure Blob Storage, converts each slide to images,
        extracts information using LLMs, and creates index records for each slide with text content,
        summaries, and key insights.

        Args:
            storage_metadata (StorageMetadata): Contains blob URI and file metadata
                for the PPTX document to process.
            text_splitter (Any): Text splitting utility.
            datasource_client (AzureBlobServiceClient): Azure Blob Storage client
                for downloading source files and uploading processed images.

        Returns:
            List[MultimodalSearchIndexRecord]: A list of search index records, one per slide.
        """
        file_buffer = None  # Initialize file_buffer for cleanup in finally block
        temp_file_path = None

        try:
            install_poppler_utils()
            install_libreoffice_dependencies()
        except RuntimeError as e:
            logger.error(f"Installation failed: {e}")
            raise

        try:
            blob_uri = storage_metadata.blob_uri
            blob_uri = unquote(blob_uri)  # Decode any URL-encoded characters

            # Step 1: Download the blob content.
            logger.info("[Step 1] Downloading the blob content.")
            file_buffer = datasource_client.download_blob(blob_uri)
            metadata = datasource_client.get_blob_metadata_from_uri(blob_uri)

            # Step 2: Write the file buffer to a temporary file.
            logger.info("[Step 2] Writing the file buffer to a temporary file.")
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=FileExtensions.PPTX
            ) as temp_file:
                temp_file.write(file_buffer.read())
                temp_file_path = temp_file.name

                # Step 3: Convert pptx to image bytes.
                logger.info("[Step 3] Converting pptx files to image bytes.")
                base64_image_data = convert_pptx_to_base64_image_data(temp_file_path)

            # Step 4: Extract information from slides.
            logger.info("[Step 4] Extracting information from slides (image bytes).")
            with get_openai_callback() as cb:
                synopsis = await extract_information_from_base64_image(
                    base64_image=base64_image_data[0],
                    prompt=INTRODUCTION_SLIDE_PROMPT,
                )
                records = await self._process_all_slides(
                    base64_image_data=base64_image_data,
                    synopsis=synopsis,
                    parent_path=blob_uri,
                    datasource_client=datasource_client,
                    metadata=metadata,
                )
                self._total_cost += cb.total_cost
                self._total_tokens += cb.total_tokens
                self._prompt_tokens += cb.prompt_tokens
                self._completion_tokens += cb.completion_tokens
            return records

        except Exception as e:
            # Log error if needed and raise a descriptive error.
            logger.error(f"Failed to process the PDF document: {str(e)}")
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
                    logger.error(f"Failed to delete temporary file: {cleanup_error}")

    @staticmethod
    def _get_media_path(
        parent_path: str,
        slide_index: int,
    ) -> str:
        """Generate the media path for a slide image."""
        # full_media_path -> "container/top/index-artifacts/images/sub/file/slide_index.JPG"
        full_media_path_parts = "/".join(
            [generate_base_media_path(parent_path), f"{slide_index}.JPG"]
        )
        # media_path -> "top/index-artifacts/images/sub/file/slide_index.JPG"
        media_path = "/".join(full_media_path_parts.split("/")[1:])
        return media_path

    @staticmethod
    def _upload_slide_image(
        base64_image: str,
        media_path: str,
        datasource_client: AzureBlobServiceClient,
        metadata: Optional[dict] = None,
    ) -> None:
        """Upload a slide image to Azure Blob Storage."""

        image_bytes = base64.b64decode(base64_image)
        modified_timestamp = metadata.get("source_file_modified")
        if modified_timestamp:
            datasource_client.upload_blob(
                data=image_bytes,
                blob_name=media_path,
                metadata={
                    "source_file_modified": (
                        modified_timestamp.isoformat()
                        if isinstance(modified_timestamp, datetime)
                        else str(modified_timestamp)
                    ),
                },
            )
        else:
            datasource_client.upload_blob(data=image_bytes, blob_name=media_path)

    async def _process_single_slide(
        self,
        base64_image: str,
        slide_index: int,
        synopsis: str,
        parent_path: str,
        datasource_client: AzureBlobServiceClient,
        metadata: Optional[dict] = None,
    ) -> MultimodalSearchIndexRecord:
        """Process a single slide image to extract information and create a search index record."""

        # 1. Extract information from the slide image
        logger.info(f"Processing slide number: [{slide_index}].")
        async with self._semaphore:
            try:
                logger.info("Extracting information from the slide.")
                prompts = [SUMMARY_PROMPT, KEYINSIGHTS_PROMPT, TEXTEXTRACTION_PROMPT]
                tasks = [
                    extract_information_from_base64_image(base64_image, prompt)
                    for prompt in prompts
                ]
                summary, key_insights, extracted_text = await asyncio.gather(*tasks)
            except Exception as e:
                logger.error(f"Error extracting information from image: {e}")
                raise e

        if slide_index == 1:
            logger.info(f"Creating chunk for the introduction slide ({slide_index}).")
            chunk = (
                f"Summary: {summary}\n"
                f"Key Insights: {key_insights}\n"
                f"Extracted Text: {extracted_text}"
            ).strip()
        else:
            logger.info(f"Creating chunk for a non-introduction slide ({slide_index}).")
            chunk = (
                f"Synopsis: {synopsis}\n"
                f"Summary: {summary}\n"
                f"Key Insights: {key_insights}\n"
                f"Extracted Text: {extracted_text}"
            ).strip()

        # 2. generate the media path for the slide image
        media_path = self._get_media_path(
            parent_path=parent_path,
            slide_index=slide_index,
        )

        # 3. Upload the slide image to Azure Blob Storage
        if self._upload_image:
            try:
                self._upload_slide_image(
                    base64_image=base64_image,
                    media_path=media_path,
                    datasource_client=datasource_client,
                    metadata=metadata,
                )
                logger.info(f"Slide image {slide_index} uploaded to {media_path}.")
            except Exception as e:
                logger.error(
                    f"Error uploading slide image {slide_index} to {media_path}: {e}"
                )
                raise e
        else:
            logger.info(
                f"TEST mode: Skipping upload of slide image {slide_index} to {media_path}."
            )

        return MultimodalSearchIndexRecord(
            parent_path=parent_path,
            use_case="multimodal",
            content=chunk,
            chunk_num=1,
            chunk_hash=generate_hash(chunk),
            row=slide_index,
            sheet_name=parse_blob_uri_root_directory_name(parent_path),
            source_modality=SourceModality.IMAGE,
            media_path=media_path,
        )

    async def _process_all_slides(
        self,
        base64_image_data: List[str],
        synopsis: str,
        parent_path: str,
        datasource_client: AzureBlobServiceClient,
        metadata: Optional[dict] = None,
    ) -> List[MultimodalSearchIndexRecord]:
        """Process all slides concurrently and return a list of search index records."""

        tasks = []
        for idx, base64_image in enumerate(base64_image_data, start=1):
            tasks.append(
                self._process_single_slide(
                    base64_image=base64_image,
                    slide_index=idx,
                    synopsis=synopsis,
                    parent_path=parent_path,
                    datasource_client=datasource_client,
                    metadata=metadata,
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        records = []
        for idx, result in enumerate(results, start=1):
            if isinstance(result, Exception):
                logger.error(f"Error processing slide {idx}: {result}")
            else:
                records.append(result)
        logger.info(
            f"Processed {len(records)} slides successfully. Failed slides: {[i+1 for i, r in enumerate(results) if isinstance(r, Exception)]}"
        )
        return records
