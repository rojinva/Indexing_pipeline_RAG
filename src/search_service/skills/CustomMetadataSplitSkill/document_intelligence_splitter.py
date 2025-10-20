import os
from urllib.parse import unquote
from .text_splitter import split_text_into_chunks
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .excel_splitter import generate_hash
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def create_document_analysis_client(endpoint, key):
    document_analysis_client = DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )
    return document_analysis_client

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def split_word_documents_into_chunks(blob_uri, chunk_size, chunk_overlap, semaphore):
    # Acquire the semaphore
    async with semaphore:
        try:
            unquoted_blob_uri = unquote(blob_uri)
            parent_path_with_container_name = unquoted_blob_uri.split(
                ".blob.core.windows.net/"
            )[1]
            use_case = parent_path_with_container_name.split("/")[1]
            endpoint = os.environ["DOCUMENT_INTELLIGENCE_ENDPOINT"]
            key = os.environ["DOCUMENT_INTELLIGENCE_KEY"]
            document_analysis_client = await create_document_analysis_client(endpoint, key)

            poller = await document_analysis_client.begin_analyze_document(
                "prebuilt-layout",
                {"urlSource": unquoted_blob_uri},
                output_content_format="markdown",
            )
            result = await poller.result()
            docs_string = result.content

            headers_to_split_on = [("#", "Section")]

            text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, strip_headers=False
            )
            md_header_splits = text_splitter.split_text(docs_string)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

            splits = text_splitter.split_documents(md_header_splits)

            chunks_with_metadata = []

            for split in splits:
                text = split.page_content
                sheet_name = split.metadata.__str__()

                data = {
                    "content": text,
                    "chunk_hash": generate_hash(text),
                    "use_case": use_case,
                    "sheet_name": sheet_name,
                    "row": -1,
                    "parent_path": parent_path_with_container_name,
                }

                chunks_with_metadata.append(data)

            return chunks_with_metadata

        except Exception as e:
            return f"{str(e)}."
        finally:
            await document_analysis_client.close()

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def split_powerpoint_pdf_into_chunks(blob_uri, chunk_size, chunk_overlap, semaphore):
    # Acquire the semaphore
    async with semaphore:
        try:
            unquoted_blob_uri = unquote(blob_uri)
            parent_path_with_container_name = unquoted_blob_uri.split(
                ".blob.core.windows.net/"
            )[1]
            use_case = parent_path_with_container_name.split("/")[1]
            endpoint = os.environ["DOCUMENT_INTELLIGENCE_ENDPOINT"]
            key = os.environ["DOCUMENT_INTELLIGENCE_KEY"]
            document_analysis_client = await create_document_analysis_client(endpoint, key)

            poller = await document_analysis_client.begin_analyze_document(
                "prebuilt-layout",
                {"urlSource": unquoted_blob_uri},
                output_content_format="markdown",
            )

            result = await poller.result()
            chunks_with_metadata = []
            for page in result.pages:
                page_number = page.page_number
                if "lines" in page.keys():  # only PDFs have lines, PPTs wont have lines
                    text = " ".join(
                        [text_content_dict.content for text_content_dict in page.lines]
                    )
                else:
                    text = " ".join([line.content for line in page.words])

                chunked_text = split_text_into_chunks(
                    content=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )

                for text in chunked_text:
                    data = {
                        "content": text,
                        "chunk_hash": generate_hash(text),
                        "use_case": use_case,
                        "sheet_name": "",
                        "row": page_number,
                        "parent_path": parent_path_with_container_name,
                    }
                    chunks_with_metadata.append(data)

            return chunks_with_metadata

        except Exception as e:
            return f"{str(e)}."
        finally:
            await document_analysis_client.close()