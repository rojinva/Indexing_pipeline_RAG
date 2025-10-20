import os
from urllib.parse import unquote
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential
from .document_image_processor import process_images_in_document, extract_document_title
from .text_splitter import split_text_into_chunks
from .excel_splitter import generate_hash, get_blob_size
from .document_image_processor import convert_document_to_pdf, extract_sections_from_word_document , convert_doc_docx_to_pdf , get_file_from_adls ,convert_ppt_to_pdf
import re
import difflib
import time
from .fabric_utils import get_file_content_from_lakehouse
from urllib.parse import urlparse, unquote
from PyPDF2 import PdfReader
from .constants import blob_uri_prefix_constant
import tempfile
import asyncio
from io import BytesIO
from io import BytesIO

## Helper functions
def extract_onelake_params(url: str) -> dict:
    """
    Extracts workspace_name, lakehouse_name, remote_path, and filename
    from a OneLake DFS URLam

    Args:
        url: The OneLake URL string.

    Returns:
        A dictionary containing the extracted parameters, or None if the URL
        format is invalid.
    """
    try:
        unquoted_url = unquote(url)
        parsed_url = urlparse(unquoted_url)

        if not parsed_url.netloc.endswith(".dfs.fabric.microsoft.com"):
            raise ValueError("URL does not seem to be a valid OneLake DFS URL.")

        # Path starts with '/', remove it for splitting
        path_parts = parsed_url.path.strip('/').split('/')

        if len(path_parts) < 4 or path_parts[2].lower() != 'files':
            raise ValueError("Invalid OneLake path structure. Expected '/workspace/lakehouse.Lakehouse/Files/...'")

        workspace_name = path_parts[0]
        # Remove .Lakehouse suffix if present
        lakehouse_name = path_parts[1].replace(".Lakehouse", "")
        filename = path_parts[-1]
        # Join parts between 'Files' (index 2) and the filename (last index)
        remote_path_parts = path_parts[3:-1]
        remote_path = "/".join(remote_path_parts) if remote_path_parts else ""

        return {
            "workspace_name": workspace_name,
            "lakehouse_name": lakehouse_name,
            "remote_path": remote_path,
            "filename": filename
        }
    except (ValueError, IndexError) as e:
        print(f"Error parsing URL '{url}': {e}")
        return {}
def normalize_whitespace(text):
    """
    Normalizes whitespace in the text by replacing newline characters with spaces.
    Args:
        text (str): The input text to normalize.
    Returns:
        str: The normalized text.
    """
    return re.sub(r'\s+', ' ', text).strip()    
              
def find_page_number_for_split(page_text, page_wise_text, section_name):
    normalized_page_text = normalize_whitespace(page_text)
    try:
        for page_num, text in page_wise_text.items():
            normalized_text = normalize_whitespace(text)
            if normalized_page_text[:50] in normalized_text:
                return page_num
            elif section_name in normalized_text:
                return page_num
            else:
                # Use difflib to find partial matches
                close_matches = difflib.get_close_matches(normalized_page_text[:50], [normalized_text], n=1, cutoff=0.5)
                if close_matches:
                    return page_num
        return -1
    except Exception as e:
        print("Cannot retrieve page number due to no text, probably empty page or has an image : ",str(e))
        return -1

def extract_section_or_subsection(text, priority='Section'):
    """
    Extracts the value associated with 'Subsection' if it exists, otherwise the value associated with 'Section'.
    Args:
        text (str): The input string containing 'Section' and 'Subsection'.
    Returns:
        str: The extracted value.
    """
    section_pattern = re.compile(r"'Section':\s*'([^']*)'")
    subsection_pattern = re.compile(r"'Subsection':\s*'([^']*)'")
    if priority == 'Subsection':
        subsection_match = subsection_pattern.search(text)
        if subsection_match:
            return subsection_match.group(1)

        section_match = section_pattern.search(text)
        if section_match:
            return section_match.group(1)
    else:
        section_match = section_pattern.search(text)
        if section_match:
            return section_match.group(1)

        subsection_match = subsection_pattern.search(text)
        if subsection_match:
            return subsection_match.group(1)
    return text

def extract_section_keys(markdown_text):
    sections = {}
    lines = markdown_text.strip().split('\n')
    for line in lines:
        if line.startswith('#'):
            # Extract the section title and its markdown format
            section_title = line.lstrip('#').strip()
            sections[section_title] = line
    return sections

# Function to replace sections with markdown format
def replace_sections(text, sections):
    for key, value in sections.items():
        # Use regex to find the section and replace it with the markdown format
        text = re.sub(rf'(?<!#)\b{re.escape(key)}\b', value, text, count=1)
    return text

def strip_markdown_headers(markdown_text):
    """
    Strips off # characters from the beginning of each line in a markdown string.
    Only those # characters that appear before the first whitespace character on each line are removed.

    Args:
        markdown_text (str): The markdown text to process.

    Returns:
        str: The processed markdown text with # characters removed from the beginning of each line.
    """
    # Use regular expression to match lines starting with # characters followed by whitespace
    # processed_text = re.sub(r'^\s*#+(?=\s)', '', markdown_text, flags=re.MULTILINE)
    processed_text = re.sub(r'^\s*#+\s*', '', markdown_text, flags=re.MULTILINE)
    return processed_text

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
                blob_uri_prefix_constant
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
                blob_uri_prefix_constant
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


async def split_ppt_pdf_lakehouse(blob_uri, chunk_size, chunk_overlap, semaphore):
    # Acquire the semaphore
    async with semaphore:
        try:
            unquoted_blob_uri = unquote(blob_uri)
            parent_path_with_container_name = unquoted_blob_uri.split(
                "https://onelake.dfs.fabric.microsoft.com/"
            )[-1]
            print("Parent path with container name: ", parent_path_with_container_name)
            use_case = parent_path_with_container_name.split("/")[3]
            print("Use case: ", use_case)
            params = extract_onelake_params(blob_uri)
            workspce_name = params["workspace_name"]
            lakehouse_name = params["lakehouse_name"]
            remote_path = params["remote_path"]
            filename = params["filename"]
            endpoint = os.environ["DOCUMENT_INTELLIGENCE_ENDPOINT"]
            key = os.environ["DOCUMENT_INTELLIGENCE_KEY"]
            document_analysis_client = await create_document_analysis_client(endpoint, key)
            print("Document analysis client created", document_analysis_client)
            file_content_bytes = get_file_content_from_lakehouse(
                account_name="onelake",
                workspace_name=workspce_name,
                lakehouse_name=lakehouse_name,
                remote_path=remote_path,
                filename=filename,
                tenant_id=os.environ['TENET_ID'],
                client_id=os.environ['CLIENT_ID'],
                client_secret=os.environ['CLIENT_SECRET'],
            )

            poller = await document_analysis_client.begin_analyze_document(
                "prebuilt-layout",
                analyze_request=file_content_bytes,
                content_type="application/octet-stream",
                output_content_format="markdown"
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
async def split_edms_word_documents(blob_uri, include_metadata_in_chunk, process_images, chunk_size, chunk_overlap, semaphore):
    """
    Splits a Word document into chunks with metadata using Azure Document Intelligence.
    Args:
        blob_uri (str): The URI of the blob containing the Word document.
        include_metadata_in_chunk (bool): Flag to include metadata in each chunk.
        semaphore (asyncio.Semaphore): Semaphore to control concurrency.
    Returns:
        list: A list of dictionaries, each containing a chunk of the document with metadata.
    Raises:
        Exception: If an error occurs during processing.
    The function performs the following steps:
    1. Acquires the semaphore.
    2. Processes the input blob URI.
    3. Submits a document cracking job to Azure Document Intelligence.
    4. Splits the resulting markdown content by section and subsection headers.
    5. Optionally includes metadata in each chunk.
    6. Extracts images from the Word file and generates summaries.
    7. Returns a list of chunks with metadata.
    """
    # Acquire the semaphore
    async with semaphore:
        document_analysis_client = None
        try:
            print("Blob URI: ", blob_uri)
            # Process input blob
            unquoted_blob_uri = unquote(blob_uri)
            parent_path_with_container_name = unquoted_blob_uri.split(
                blob_uri_prefix_constant
            )[1]
            use_case = parent_path_with_container_name.split("/")[1]
            # Extract the container name and blob name from the URI
            container_name, blob_name = parent_path_with_container_name.split("/", 1)
            blob_size = get_blob_size(tenant_id=os.environ['TENET_ID'], client_id=os.environ['CLIENT_ID'], client_secret=os.environ['CLIENT_SECRET'], storage_account_name = os.environ['STORAGE_ACCOUNT_NAME'], container_name=container_name, blob_name=blob_name)
            print("Blob size (MB): ", blob_size)
            endpoint = os.environ["DOCUMENT_INTELLIGENCE_ENDPOINT"]
            key = os.environ["DOCUMENT_INTELLIGENCE_KEY"]
            document_analysis_client = await create_document_analysis_client(endpoint, key)
            print("Document analysis client created")
            if blob_size <= 25:
                # submit document cracking job to azure document intelligence
                poller = await document_analysis_client.begin_analyze_document(
                    "prebuilt-layout",
                    {"urlSource": unquoted_blob_uri},
                    output_content_format="markdown",
                )
                start_time = time.time()
                result = await poller.result()
                print("Time taken to process document using ADI: ", time.time() - start_time)
                docs_string = result.content
            else:
                return None
            section_text = extract_sections_from_word_document(blob_uri)
            docs_string = strip_markdown_headers(docs_string)
            docs_string = replace_sections(docs_string, extract_section_keys(section_text))
            # split the markdown content by section and subsection headers
            headers_to_split_on = [("#", "Section"), ("##", "Subsection")]

            text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, strip_headers=False, return_each_line=False
            )
            md_header_splits = text_splitter.split_text(docs_string)
            print("Number of splits: ", len(md_header_splits))
            # performing recursive splitting of the text since doc intelligence splits the text into large chunks for some documents failing to identify the section headers correctly
            recursive_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="text-embedding-3-small", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            splits = recursive_text_splitter.split_documents(md_header_splits)

            chunks_with_metadata = []
            if include_metadata_in_chunk:
                document_title = extract_document_title(md_header_splits[0].page_content)
            else:
                document_title = ""
            
            for split in splits:
                page_text = split.page_content
                # sheet name will contain the section as well as subsection headers
                print("Section NameSplit: ", split.metadata)
                sheet_name = split.metadata.__str__().strip("{}")
                sheet_name = extract_section_or_subsection(sheet_name, priority='Section')

                if include_metadata_in_chunk:
                    metadata_string = document_title + " | Section: " + sheet_name
                    page_text =  metadata_string + "\n Page Content:" + page_text

                data = {
                    "content": page_text,
                    "chunk_hash": generate_hash(page_text),
                    "use_case": use_case,
                    "sheet_name": "Section: " + ("" if sheet_name == "{}" else sheet_name),
                    "row": -1,
                    "parent_path": parent_path_with_container_name,
                }

                chunks_with_metadata.append(data)

            if not process_images:
                return chunks_with_metadata
            else:
                # extract images from word file and generate summary
                image_summary_list = process_images_in_document(document_path=blob_uri)
                
                # filter out elements where image summary is None
                image_summary_list = [image_summary for image_summary in image_summary_list if image_summary is not None]

                for image_summary in image_summary_list:
                    image_summary = "Image Summary: " + image_summary
                    if include_metadata_in_chunk:
                        image_summary = document_title + "\n" + image_summary
                    # chunking image summaries when they exceed 7k tokens since for instances when its extracting markdowns of table, its going above the token limits.
                    image_splits = recursive_text_splitter.split_text(image_summary)
                    for split in image_splits:
                        image_summary = split
                        image_chunk = {
                            "content": image_summary,
                            "chunk_hash": generate_hash(image_summary),
                            "use_case": use_case,
                            "sheet_name": "Section: Derived from Image",
                            "row": -1,
                            "parent_path": parent_path_with_container_name,
                        }
                        chunks_with_metadata.append(image_chunk)
                return chunks_with_metadata

        except Exception as e:
            return f"Error Processing word documents using split_edms_word_documents() due to: {str(e)}."
        finally:
            if document_analysis_client is not None:
                await document_analysis_client.close()

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def split_edms_pptx_documents(blob_uri, include_metadata_in_chunk, process_images, semaphore):
    """
    Splits a PPTX document into chunks with metadata using Azure Document Intelligence.
    Args:
        blob_uri (str): The URI of the blob containing the PPTX document.
        include_metadata_in_chunk (bool): Flag to include metadata in each chunk.
        semaphore (asyncio.Semaphore): Semaphore to control concurrency.
    Returns:
        list: A list of dictionaries, each containing chunked content and associated metadata.
    Raises:
        Exception: If there is an error during processing.
    Notes:
        - The function acquires a semaphore to control concurrency.
        - It processes the input blob URI and extracts relevant metadata.
        - Submits a document cracking job to Azure Document Intelligence.
        - Extracts text and metadata from each page of the document.
        - Optionally includes metadata in each chunk.
        - Processes images in the document and generates summaries.
        - Returns a list of chunks with metadata.
    """
    # Acquire the semaphore
    async with semaphore:
        document_analysis_client = None
        try:
            # Process input blob
            unquoted_blob_uri = unquote(blob_uri)
            parent_path_with_container_name = unquoted_blob_uri.split(
                blob_uri_prefix_constant
            )[1]
            use_case = parent_path_with_container_name.split("/")[1]
            
            # submit document cracking job to azure document intelligence
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

                if include_metadata_in_chunk:
                    if page_number == 1:
                        document_title = extract_document_title(document_text=text)
                    text = document_title + " | Slide Content: " + text

                data = {
                    "content": text,
                    "chunk_hash": generate_hash(text),
                    "use_case": use_case,
                    "sheet_name": "",
                    "row": page_number,
                    "parent_path": parent_path_with_container_name,
                }
                chunks_with_metadata.append(data)

            if not process_images:
                return chunks_with_metadata
            else:
                # extract images from word file and generate summary
                image_summary_list = process_images_in_document(document_path=blob_uri, document_content=chunks_with_metadata)
                # the list of dictionary already has content and row fields populated
                print("Image summary list length: ", len(image_summary_list))
                for image_summary in image_summary_list:
                    image_summary['chunk_hash'] = generate_hash(image_summary["content"])
                    image_summary['use_case'] = use_case
                    image_summary['sheet_name'] = ""
                    image_summary['parent_path'] = parent_path_with_container_name

                    if include_metadata_in_chunk:
                        image_summary["content"] = document_title + "\n" + image_summary["content"]
                    
                    chunks_with_metadata.append(image_summary)

                return chunks_with_metadata

        except Exception as e:
            return f"Error in processing split_edms_pptx_documents function: {str(e)}."
        finally:
            if document_analysis_client is not None:
                await document_analysis_client.close()


async def split_edms_pdf_documents(blob_uri, include_metadata_in_chunk, process_images, chunk_size, chunk_overlap, semaphore):
    async with semaphore:
        document_analysis_client = None
        blob_uri_prefix_constant = ".blob.core.windows.net/"
        temp_pdf_path = None
        try:
            # Process input blob
            unquoted_blob_uri = unquote(blob_uri)
            parent_path_with_container_name = unquoted_blob_uri.split(blob_uri_prefix_constant)[1]
            use_case = parent_path_with_container_name.split("/")[1]
            file_extension = unquoted_blob_uri.split(".")[-1]
            print("File extension: ", file_extension)

            # submit document cracking job to azure document intelligence
            endpoint = os.environ["DOCUMENT_INTELLIGENCE_ENDPOINT"]
            key = os.environ["DOCUMENT_INTELLIGENCE_KEY"]

            document_analysis_client = await create_document_analysis_client(endpoint, key)
            if file_extension != "pdf":
                temp_pdf_path = convert_document_to_pdf(unquoted_blob_uri)
                with open(temp_pdf_path, "rb", encoding="utf-8") as f:
                    poller = document_analysis_client.begin_analyze_document(
                            "prebuilt-layout",
                            analyze_request=f,
                            content_type="application/octet-stream",
                            output_content_format="markdown"
                        )
                    result = poller.result()
                print("File is not a PDF")
                pass
            else:
                # temp_pdf_path = unquoted_blob_uri
                file_bytes = get_file_from_adls(unquoted_blob_uri)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(file_bytes)
                    temp_pdf_path = temp_pdf.name

                poller = await document_analysis_client.begin_analyze_document(
                    "prebuilt-layout",
                    {"urlSource": unquoted_blob_uri},
                    output_content_format="markdown",
                )
                result = await poller.result()

            
            # markdown content of the pdf document to be used for markdown splitting
            docs_string = result.content
            # page wise content of the pdf document
            page_wise_text = {}
            try:
                # Split markdown content per page
                page_num = 1
                for page in result.pages: 
                    content = result.content[page.spans[0]['offset']: page.spans[0]['offset'] + page.spans[0]['length']]
                    # content_per_page.append({"page_number": page_num, "content": content})
                    page_wise_text[page_num] = content
                    page_num+=1
            except Exception as e:
                print(f"Error in processing page wise text in split_edms_pdf_documents() : {str(e)}.")
            # Split the markdown content by section and subsection headers
            headers_to_split_on = [("#", "Section"), ("##", "Subsection")]
            text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
            md_header_splits = text_splitter.split_text(docs_string)
            print("Number of splits: ", len(md_header_splits))
            # Perform recursive splitting of the text
            recursive_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            splits = recursive_text_splitter.split_documents(md_header_splits)
            print("Number of splits after recursive splitting: ", len(splits))
            chunks_with_metadata = []
            
            for split in splits:
                page_text = split.page_content
                section_name = split.metadata.__str__().strip("{}")
                section_name = extract_section_or_subsection(section_name, priority='Section')
                if include_metadata_in_chunk:
                    document_title = extract_document_title(md_header_splits[0].page_content)
                    print("Extracted Document title: ", document_title)
                    metadata_string = document_title + " | Section: " + section_name
                    page_text = metadata_string + "\n Page Content:" + page_text
            
                page_number = find_page_number_for_split(page_text, page_wise_text, extract_section_or_subsection(section_name, priority='Section'))
                    
                data = {
                    "content": page_text,
                    "chunk_hash": generate_hash(page_text),
                    "use_case": use_case,
                    "sheet_name": "" if section_name == "{}" else section_name,
                    "row": page_number, # this might not yield exact page number but will represent the starting page number of the split. The chunk/split could have multiple pages text in it.
                    "parent_path": parent_path_with_container_name,
                }
                chunks_with_metadata.append(data)
            
            if not process_images:
                return chunks_with_metadata
            else:
                # Identify images in the PDF document and generate summaries

                image_summary_list = process_images_in_document(file_bytes, chunks_with_metadata)
                # image_summary_list = process_images_in_document(document_path=temp_pdf_path, document_content=chunks_with_metadata)
                image_summary_list = [image_summary for image_summary in image_summary_list if image_summary is not None]
                for image_summary in image_summary_list:
                    image_summary_content = image_summary["content"]
                    if include_metadata_in_chunk:
                        image_summary_content = document_title + "\n" + image_summary_content
                    
                    image_chunk = {
                        "content": image_summary_content,
                        "chunk_hash": generate_hash(image_summary_content),
                        "use_case": use_case,
                        "sheet_name": "",
                        "row": image_summary["row"],
                        "parent_path": parent_path_with_container_name,
                    }
                    chunks_with_metadata.append(image_chunk)

                return chunks_with_metadata

        except Exception as e:
            return f"Error in processing split_edms_pdf_documents function: {str(e)}."
        finally:
            if document_analysis_client is not None:
                await document_analysis_client.close()
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def extract_doc_pages(blob_uri, semaphore):
    """
    Extracts text from each page of a document file located at a given blob URI and converts it to PDF.

    This asynchronous function retrieves a document from Azure Blob Storage, converts it to PDF,
    and extracts text from each page using PyPDF2. The extracted text is returned along with metadata
    for each page.

    Args:
        blob_uri (str): The URI of the blob containing the document file.
        semaphore (asyncio.Semaphore): A semaphore to limit concurrent access.

    Returns:
        list: A list of dictionaries containing extracted text and metadata for each page.

    Raises:
        Exception: If there is an error during the extraction process, an exception is raised
                   with a message indicating the failure.
    """
    async with semaphore:
        try:
            # Unquote the blob URI to handle any encoded characters
            unquoted_blob_uri = unquote(blob_uri)
            
            # Extract the path and use case from the blob URI
            parent_path_with_container_name = unquoted_blob_uri.split(blob_uri_prefix_constant)[1]
            use_case = parent_path_with_container_name.split("/")[1]
            
            # Retrieve the document file from Azure Blob Storage
            blob_data = get_file_from_adls(blob_uri=blob_uri)
            
            # Convert the document to PDF
            pdf_buf = convert_doc_docx_to_pdf(blob_data)
            
            # Initialize a PDF reader to extract text from the PDF
            reader = PdfReader(pdf_buf)
            chunks_with_metadata = []
            
            # Iterate over each page in the PDF and extract text
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                
                # Append extracted text and metadata to the list
                chunks_with_metadata.append({
                    "content": text.strip(),
                    "chunk_hash": generate_hash(text.strip()),
                    "use_case": use_case,
                    "sheet_name": "",
                    "row": page_number,
                    "parent_path": parent_path_with_container_name 
                })
            
            # Return the list of extracted text and metadata
            return chunks_with_metadata
        
        except Exception as e:
            # Return the error message if an exception occurs
            return f"{str(e)}."

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def split_ppt_into_chunks(blob_uri, chunk_size: int, chunk_overlap: int, semaphore) -> list:
    """
    Asynchronously splits the content of a PPT into chunks.

    Args:
        blob_uri (str): The URI of the PPT file in the blob storage.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.
        semaphore (asyncio.Semaphore): Semaphore to control concurrency.

    Returns:
        list: A list of chunks with metadata.
    """
    async with semaphore:
        unquoted_blob_uri = unquote(blob_uri)
        blob_uri_prefix_constant = ".blob.core.windows.net/"
        parent_path_with_container_name = unquoted_blob_uri.split(
            blob_uri_prefix_constant
        )[1]
        use_case = parent_path_with_container_name.split("/")[1]

        # Fetch the file from ADLS asynchronously
        blob_data =  get_file_from_adls(blob_uri=blob_uri)

        # Convert PPT to PDF asynchronously
        pdf_buf = convert_ppt_to_pdf(blob_data)
        pdf_stream = BytesIO(pdf_buf)

        # Read the PDF content
        reader = PdfReader(pdf_stream)
        chunks_with_metadata = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()  # Extract text from the page
            if not text:
                continue

            # Split the text into chunks
            chunked_text = split_text_into_chunks(content=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            for text_chunk in chunked_text:
                data = {
                    "content": text_chunk,
                    "chunk_hash": generate_hash(text),
                    "use_case": use_case,
                    "row": page_number,
                    "parent_path": parent_path_with_container_name,
                }
                chunks_with_metadata.append(data)

        return chunks_with_metadata