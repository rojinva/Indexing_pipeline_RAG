import asyncio
import base64
import io
import os
import tempfile
from pdf2image import convert_from_path
from typing import List, Dict, Optional
from langchain_openai import AzureChatOpenAI
from langchain.schema.messages import SystemMessage, HumanMessage
from langchain_community.callbacks.manager import get_openai_callback
from ..models.records import BaseSearchIndexRecord
from ..utils.hashing import generate_hash
from ..utils.string_parsing import parse_blob_uri_root_directory_name
from ..utils.dependencies import install_poppler_utils
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


logger = logging.getLogger(__name__)

class PDFPageMarkdownProcessor:
    def __init__(
        self,
    ):
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-4o-gs",
            api_version="2024-12-01-preview",
            api_key=os.environ["AZURE_OPENAI_API_KEY_USE2"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_USE2"],
            temperature=0.0,
            max_tokens=2500,
            streaming=False,
        )
        self.total_cost = 0.0
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        Converts the PDF into a list of PIL images, one per page,
        using pdf2image.convert_from_path.
        """
        return convert_from_path(pdf_path)

    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode()
        buffered.close()
        return image_b64

    def _prepare_messages(self, image_b64: str) -> List:
        system_prompt = """
        You are a high‑precision document extractor. When given a page image, follow these rules without deviation:

        Full‑page text extraction
        • Extract every piece of text up to—but not including—any section titled “Disclosure” (case‑insensitive). As soon as you detect “Disclosure,” stop all text extraction.
        • Extract the date from the top (if present) and add it to a section named "Extracted Date" in datetime format.
        
        Table handling
        • Automatically detect all table regions.
        • Reproduce each table in Markdown, preserving every header, row label, and numeric value exactly—no rounding, reformatting, unit conversion, or omission.
        
        Graph & plot 
        • For every line graph, bar chart, scatter plot, etc., extract the full set of data points into a Markdown table.
        • Include columns for each axis (e.g. X, Y), series names, direct data labels, annotations, legend labels—verbatim as they appear.
        • Maintain numeric precision exactly. If tick marks or labels are too small or ambiguous, try to infer the values based on the graph’s scale for the relative axis.
        • Labels on the graph should be extracted and included in the Markdown table.
        • For illustrative graphs/figures (typically with data points and axis), analyze and describe them in detail and include analysis in-place where the figure appears to maintain the original layout.
        
        Output constraints
        • Output only the raw extracted content: headings, paragraphs, and Markdown tables. Do not include any AI commentary, analysis, or metadata.
        • Do not echo or reference any content from the “Disclosure” section or anything following its start.
        
        Vigilance & warnings
        • Double‑check every number, label, and annotation against the source image to ensure fidelity.
        • If any region is faint, damaged, or borderline, proceed with extreme caution rather than guess.
        • Any deviation, omission, or rounding error is unacceptable—accuracy is paramount.
        """
        user_prompt = [
            {
                "type": "text",
                "text": (
                    "Extract all text, tables, and graph data as Markdown—preserve every header, label, and number exactly."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
        ]
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        return messages

    def process(
        self, 
        storage_metadata,
        datasource_client, 
        text_splitter,
        output_path: Optional[str] = None
    ) -> List[BaseSearchIndexRecord]:
        try:
            install_poppler_utils()
        except RuntimeError as e:
            logger.error(f"Poppler Utils installation failed: {e}")
            raise
        # Download the blob content using the datasource_client
        file_buffer = datasource_client.download_blob(storage_metadata.blob_uri)
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_buffer.read())
                temp_file_path = temp_file.name

            # Convert PDF to images
            images = self._pdf_to_images(temp_file_path)
            records = []
            for i, image in enumerate(images):
                print(f"Processing page {i + 1}/{len(images)}...")
                image_b64 = self._image_to_base64(image)
                image.close()
                messages = self._prepare_messages(image_b64)
                with get_openai_callback() as cb:
                    response = self.llm.invoke(messages)
                    self.total_cost += cb.total_cost
                    self.total_tokens += cb.total_tokens
                    self.prompt_tokens += cb.prompt_tokens
                    self.completion_tokens += cb.completion_tokens

                # Use the text splitter if provided
                chunked_contents = [response.content]
                if text_splitter:
                    chunked_contents = text_splitter.split_text(response.content)

                # Skip chunks that are empty or contain only whitespace
                if not content_chunk.strip():
                    logger.warning(f"Skipping empty chunk {chunk_index} on page {i + 1}.")
                    continue
                
                for chunk_index, content_chunk in enumerate(chunked_contents, start=1):
                    record = BaseSearchIndexRecord(
                        parent_path=storage_metadata.blob_uri,
                        use_case="PDFPageMarkdownProcessor",
                        content=content_chunk,
                        chunk_num=chunk_index,
                        chunk_hash=generate_hash(content_chunk),
                        row=i + 1,
                        sheet_name=parse_blob_uri_root_directory_name(storage_metadata.blob_uri),
                    )
                    records.append(record)
            
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    for page in records:
                        f.write(f"<!-- Page {page.row}, Chunk {page.chunk_num} -->\n")
                        f.write(page.content)
                        f.write("\n\n---\n\n")
            return records
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    print(f"Failed to delete temporary file: {e}")

    async def aprocess(
        self, 
        storage_metadata,
        datasource_client, 
        text_splitter,
        output_path: Optional[str] = None
    ) -> List[BaseSearchIndexRecord]:
        try:
            install_poppler_utils()
        except RuntimeError as e:
            logger.error(f"Poppler Utils installation failed: {e}")
            raise
        # Download the blob content using the datasource_client
        file_buffer = datasource_client.download_blob(storage_metadata.blob_uri)
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_buffer.read())
                temp_file_path = temp_file.name

            images = self._pdf_to_images(temp_file_path)

            async def process_image(i: int, image: Image.Image):
                logger.info(f"Processing page {i + 1}/{len(images)}...")
                image_b64 = self._image_to_base64(image)
                image.close()
                messages = self._prepare_messages(image_b64)
                with get_openai_callback() as cb:
                    response = await self.llm.ainvoke(messages)
                    self.total_cost += cb.total_cost
                    self.total_tokens += cb.total_tokens
                    self.prompt_tokens += cb.prompt_tokens
                    self.completion_tokens += cb.completion_tokens
                logger.info(f"Page {i + 1} processed successfully.")

                # Use the text splitter if provided
                chunked_contents = [response.content]
                if text_splitter:
                    logger.info("Splitting text into chunks using the provided text splitter.")
                    chunked_contents = text_splitter.split_text(response.content)

                records = []
                for chunk_index, content_chunk in enumerate(chunked_contents, start=1):
                    logger.info(f"Creating record for chunk {chunk_index} on page {i + 1}.")
                    
                    # Skip chunks that are empty or contain only whitespace
                    if not content_chunk.strip():
                        logger.warning(f"Skipping empty chunk {chunk_index} on page {i + 1}.")
                        continue

                    record = BaseSearchIndexRecord(
                        parent_path=storage_metadata.blob_uri,
                        use_case="PDFPageMarkdownProcessor",
                        content=content_chunk,
                        chunk_num=chunk_index,
                        chunk_hash=generate_hash(content_chunk),
                        row=i + 1,
                        sheet_name=parse_blob_uri_root_directory_name(storage_metadata.blob_uri),
                    )
                    records.append(record)
                logger.info(f"Page {i + 1} processed with {len(records)} records.")
                return records

            tasks = [process_image(i, image) for i, image in enumerate(images)]
            results = await asyncio.gather(*tasks)
            # Flatten the list of lists
            records = [record for sublist in results for record in sublist]
            
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    for record in records:
                        f.write(f"<!-- Page {record.row}, Chunk {record.chunk_num} -->\n")
                        f.write(record.content)
                        f.write("\n\n---\n\n")
            return records
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    print(f"Failed to delete temporary file: {e}")

    def get_cost_estimate(self):
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }