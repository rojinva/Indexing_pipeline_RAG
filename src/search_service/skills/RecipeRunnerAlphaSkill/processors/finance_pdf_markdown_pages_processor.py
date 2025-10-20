from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from ..models.records import FinanceSearchIndexRecord
from .pdf_markdown_pages_processor import PDFPageMarkdownProcessor
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class Metadata(BaseModel):
    """Extracting key information from the document."""

    title: str = Field(
        ...,
        description="The official title of the document, typically not the first line but a key identifier of its content.",
    )
    contributing_authors: List[str] = Field(
        ...,
        description="A list of all authors who contributed to the creation of the document.",
    )
    publication_date: datetime = Field(
        ...,
        description="The date when the document was published or officially created.",
    )


class FinancePDFMarkdownPagesProcessor(PDFPageMarkdownProcessor):
    @staticmethod
    def _get_company_name(url: str) -> str:
        """
        Extracts the company name from the given Azure Blob Storage URL based on specific categories.

        Args:
            url (str): The Azure Blob Storage URL.

        Returns:
            str: The company name extracted from the URL, or an empty string if not found.
        """
        # Parse the URL and decode the path to handle characters like %20
        parsed_url = urllib.parse.urlparse(url)
        decoded_path = urllib.parse.unquote(parsed_url.path)

        # Split the path into segments and filter out empty segments.
        segments = [seg for seg in decoded_path.split("/") if seg]

        if not segments:
            return ""

        # Determine the category based on the second segment
        if len(segments) > 1:
            category = segments[1].lower()

            # Logic for "IR - Analyst Reports"
            if "ir - analyst reports" in category:
                # Company name is likely in the fourth segment
                if len(segments) > 3:
                    return segments[3].replace("%20", " ")  # Replace encoded spaces with actual spaces

            # Logic for "Competitive Analysis Events"
            elif "competitive analysis events" in category:
                # Company name is likely in the third segment
                if len(segments) > 2:
                    return segments[2].replace("%20", " ")  # Replace encoded spaces with actual spaces

        # If no match is found, return an empty string
        return ""

    @retry(
        stop=stop_after_attempt(5),  # Retry up to 5 times
        wait=wait_exponential(
            multiplier=1, min=1, max=10
        ),  # Exponential backoff: 1s, 2s, 4s, etc.
        retry=retry_if_exception_type(Exception),  # Retry on any exception
    )
    def extract_metadata(self, content: str) -> tuple:
        """
        Extracts metadata (title, contributing authors, publication date) from the given content using the structured LLM.

        Args:
            content (str): The text content from which metadata needs to be extracted.

        Returns:
            tuple: A tuple containing the title (str), contributing authors (str), and publication date (str in ISO format).
        """
        try:
            logger.info("Extracting metadata from content...")
            # Invoke the structured LLM to extract metadata
            structured_llm = self.llm.with_structured_output(Metadata)
            info = structured_llm.invoke(content)
            logger.info("Metadata extraction successful.")

            # Extract title
            title = info.title

            # Extract contributing authors and join them into a single string
            contributing_authors = ", ".join(info.contributing_authors)

            # Extract publication date and convert it to ISO format in UTC timezone
            publication_date = info.publication_date.astimezone(
                timezone.utc
            ).isoformat()

            return title, contributing_authors, publication_date
        except Exception as e:
            raise RuntimeError(f"Failed to extract metadata: {str(e)}")

    def process(
        self,
        storage_metadata,
        datasource_client,
        text_splitter=None,
        output_path: Optional[str] = None,
    ) -> List[FinanceSearchIndexRecord]:
        """
        Processes .docx files synchronously, with custom transformations applied to the text.
        :return: List of FinanceSearchIndexRecord objects.
        """
        # Call the parent class's process method
        base_records = super().process(
            storage_metadata, datasource_client, text_splitter, output_path
        )

        # Extract metadata from the first record
        if not base_records:
            raise ValueError("No records found to process.")

        record_content = base_records[0].content + "\n\n" + base_records[1].content + "\n\n" + base_records[2].content + "\n\n" + base_records[3].content
        title, contributing_authors, publication_date = self.extract_metadata(
            record_content
        )

        # Transform BaseSearchIndexRecord into FinanceSearchIndexRecord
        finance_records = []
        for base_record in base_records:
            finance_record = FinanceSearchIndexRecord(
                parent_path=base_record.parent_path,
                use_case=base_record.use_case,
                content=base_record.content,
                chunk_num=base_record.chunk_num,
                chunk_hash=base_record.chunk_hash,
                row=base_record.row,
                sheet_name=base_record.sheet_name,
                title=title,
                contributing_authors=contributing_authors,
                publication_date=publication_date,
                company_name=self._get_company_name(
                    base_record.parent_path
                ),  # Extract company name from URL
            )
            finance_records.append(finance_record)

        return finance_records

    async def aprocess(
        self,
        storage_metadata,
        datasource_client,
        text_splitter=None,
        output_path: Optional[str] = None,
    ) -> List[FinanceSearchIndexRecord]:
        """
        Processes .docx files asynchronously, with custom transformations applied to the text.
        :return: List of FinanceSearchIndexRecord objects.
        """
        # Call the parent class's aprocess method
        base_records = await super().aprocess(
            storage_metadata, datasource_client, text_splitter, output_path
        )

        # Extract metadata from the first record
        if not base_records:
            raise ValueError("No records found to process.")
        
        logger.info("Extracting metadata from the first record content...")
        record_content = base_records[0].content + "\n\n" + base_records[1].content + "\n\n" + base_records[2].content + "\n\n" + base_records[3].content
        title, contributing_authors, publication_date = self.extract_metadata(
            record_content
        )
        logger.info("Metadata extraction completed successfully.")
        # Transform BaseSearchIndexRecord into FinanceSearchIndexRecord
        finance_records = []
        for base_record in base_records:
            finance_record = FinanceSearchIndexRecord(
                parent_path=base_record.parent_path,
                use_case=base_record.use_case,
                content=base_record.content,
                chunk_num=base_record.chunk_num,
                chunk_hash=base_record.chunk_hash,
                row=base_record.row,
                sheet_name=base_record.sheet_name,
                title=title,
                contributing_authors=contributing_authors,
                publication_date=publication_date,
                company_name=self._get_company_name(
                    base_record.parent_path
                ),  # Extract company name from URL
            )
            finance_records.append(finance_record)

        return finance_records
