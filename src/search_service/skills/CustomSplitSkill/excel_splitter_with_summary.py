import os
import re
from typing import List, Dict, Any, TypedDict, Union
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from io import BytesIO

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Excel and Markdown processing
import pandas as pd
from markitdown import MarkItDown

from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv, find_dotenv
from urllib.parse import unquote
import io
from tenacity import retry, stop_after_attempt, wait_exponential
from .constants import blob_uri_prefix_constant
from .excel_splitter import get_blob_service_client, get_blob_filebuffer_and_metadata, generate_hash


load_dotenv(
    find_dotenv()
)


class WorkflowState(TypedDict):
    excel_file_path: Union[str, BytesIO]
    excel_data: Dict[str, str]
    summaries: Dict[str, str]
    chunks: List[Dict[str, Any]]
    error: str

    
    
class ExcelToMarkdownProcessor:
    """Main processor class for Excel to Markdown conversion with LangGraph orchestration"""
    
    def __init__(self, 
                 azure_openai_api_key: str,
                 azure_openai_endpoint: str,
                 azure_openai_api_version: str = "2024-02-15-preview",
                 deployment_name: str = "gpt-4o",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the processor with Azure OpenAI configuration
        
        Args:
            azure_openai_api_key: Azure OpenAI API key
            azure_openai_endpoint: Azure OpenAI endpoint URL
            azure_openai_api_version: API version
            deployment_name: GPT-4o deployment name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.azure_llm = AzureChatOpenAI(
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            api_version=azure_openai_api_version,
            deployment_name=deployment_name,
            temperature=0.1
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name="gpt-4"
        )
        
        self.markitdown = MarkItDown()
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("load_excel", self._load_excel_node)
        workflow.add_node("convert_to_markdown", self._convert_to_markdown_node)
        workflow.add_node("generate_summaries", self._generate_summaries_node)
        workflow.add_node("create_chunks", self._create_chunks_node)
        
        # Define the flow
        workflow.set_entry_point("load_excel")
        workflow.add_edge("load_excel", "convert_to_markdown")
        workflow.add_edge("convert_to_markdown", "generate_summaries")
        workflow.add_edge("generate_summaries", "create_chunks")
        workflow.add_edge("create_chunks", END)
        
        return workflow.compile()
    
    async def _load_excel_node(self, state: WorkflowState) -> WorkflowState:
        """Node to load Excel file and extract sheet information"""
        try:
            excel_file_path = state["excel_file_path"]
            
            # Read Excel file to get all sheet names
            excel_file = pd.ExcelFile(excel_file_path)
            sheet_names = excel_file.sheet_names
            
            # Initialize excel_data dictionary
            state["excel_data"] = {sheet_name: "" for sheet_name in sheet_names}
            
            print(f"[INFO] ----> Loaded Excel file with {len(sheet_names)} sheets: {sheet_names}")
            
        except Exception as e:
            state["error"] = f"Error loading Excel file: {str(e)}"
            print(f"[INFO] ----> Error loading Excel: {e}")
        
        return state
    
    async def _convert_to_markdown_node(self, state: WorkflowState) -> WorkflowState:
        """Node to convert Excel sheets to Markdown using markitdown and pandas"""
        try:
            excel_file_input = state["excel_file_path"]
            is_buffer = isinstance(excel_file_input, BytesIO)

            if is_buffer:
                excel_file_input.seek(0)
            result = self.markitdown.convert(excel_file_input)
            full_markdown = result.text_content

            if is_buffer:
                excel_file_input.seek(0)
                excel_file = pd.ExcelFile(excel_file_input, engine="openpyxl")
            else:
                excel_file = pd.ExcelFile(excel_file_input, engine="openpyxl")

            excel_data = {}
            for sheet_name in excel_file.sheet_names:
                if is_buffer:
                    excel_file_input.seek(0)
                df = pd.read_excel(excel_file_input, sheet_name=sheet_name, engine="openpyxl")
                sheet_markdown = f"# Sheet: {sheet_name}\n\n"
                sheet_markdown += df.to_markdown(index=False)
                sheet_markdown += "\n\n---\n\n"
                excel_data[sheet_name] = sheet_markdown

            state["excel_data"] = excel_data
            print(f"[INFO] ----> Converted {len(excel_data)} sheets to Markdown")

        except Exception as e:
            state["error"] = f"Error converting to Markdown: {str(e)}"
            print(f"[INFO] ----> Error converting to Markdown: {e}")
        return state

    async def _generate_summaries_node(self, state: WorkflowState) -> WorkflowState:
        """Node to generate summaries using Azure OpenAI GPT-4o"""
        try:
            excel_data = state["excel_data"]
            summaries = {}
            
            for sheet_name, markdown_content in excel_data.items():
                # Create summarization prompt
                prompt = f"""
                Please provide a professional and concise to the point summary of the 
                following Excel sheet data in markdown format. Always try to focus on the 
                key insights, patterns, and important information.
                
                Sheet Name: {sheet_name}
                
                Content:
                {markdown_content}
                
                Summary:
                """
                
                # Generate summary using Azure OpenAI
                response = await self.azure_llm.ainvoke(prompt)
                summary = response.content.strip()
                
                summaries[sheet_name] = summary
                print(f"[INFO] ----> Generated summary for sheet: {sheet_name}")
            
            state["summaries"] = summaries
            print(f"[INFO] ----> Generated summaries for {len(summaries)} sheets")
            
        except Exception as e:
            state["error"] = f"Error generating summaries: {str(e)}"
            print(f"[INFO] ----> Error generating summaries: {e}")
        
        return state
    
    async def _create_chunks_node(self, state: WorkflowState) -> WorkflowState:
        """Node to create chunks with metadata"""
        try:
            excel_data = state["excel_data"]
            summaries = state["summaries"]
            excel_file_path = state["excel_file_path"]
            
            chunks = []
            
            # Regular expression to match NaN, NA, nan, Nan
            nan_pattern = re.compile(r'\b(?:NaN|NA|nan|Nan|NaT|NAT|na|Na|NAN)\b', re.IGNORECASE)
            
            for sheet_name, markdown_content in excel_data.items():
                # Split content into chunks
                documents = self.text_splitter.create_documents([markdown_content])
                
                # Create JSON objects for each chunk
                for i, doc in enumerate(documents):
                    # Clean the content by removing NaN, NA, nan, Nan
                    cleaned_content = nan_pattern.sub('', doc.page_content)
                    
                    chunk_data = {
                        "chunks": cleaned_content,
                        "chunks_hash" : generate_hash(cleaned_content),
                        "sheet_summary": summaries.get(sheet_name, ""),
                        "sheet_name": sheet_name,
                        "usecase" : "LamIndiaFinance&Travel",
                        "row" : -1,
                    }
                    chunks.append(chunk_data)
                
                print(f"[INFO] ----> Created {len(documents)} chunks for sheet: {sheet_name}")
            
            state["chunks"] = chunks
            print(f"[INFO] ----> Total chunks created: {len(chunks)}")
            
        except Exception as e:
            state["error"] = f"Error creating chunks: {str(e)}"
            print(f"[INFO] ----> Error creating chunks: {e}")
        
        return state
    
    
    async def process_excel_file(self, excel_file_path: str) -> List[Dict[str, Any]]:
        """
        Main function to process Excel file through the LangGraph workflow
        
        Args:
            excel_file_path: Path to the Excel file
            
        Returns:
            List of JSON objects containing chunks with metadata
        """
        # Initialize state
        initial_state = WorkflowState(
            excel_file_path=excel_file_path,
            excel_data={},
            summaries={},
            chunks=[],
            error=""
        )
        
        print(f"[INFO] ----> Starting Excel processing workflow for: {excel_file_path}")
        
        # Run the workflow
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            
            if final_state.get("error"):
                raise Exception(final_state["error"])
            
            print(f"[INFO] ----> Workflow completed successfully!")
            print(f"[INFO] ----> Processed {len(final_state['excel_data'])} sheets")
            print(f"[INFO] ----> Generated {len(final_state['summaries'])} summaries")
            print(f"[INFO] ----> Created {len(final_state['chunks'])} chunks")
            
            return final_state["chunks"]
            
        except Exception as e:
            print(f"[INFO] ----> Workflow failed: {e}")
            raise e


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generateExcelChunksWithSummary(blob_uri : str = None, chunk_size : int = 2048, chunk_overlap : int = 512):

    print("[INFO] ----> Execution on Azure Blob file.....")
    print("[INFO] ----> Please wait....\n")
    
    file_buffer, data_path, _, _, _ = get_blob_filebuffer_and_metadata(
        blob_uri=blob_uri
    )

    AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_SUMMARIZATION_MODEL")
    AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")


    processor = ExcelToMarkdownProcessor(
        azure_openai_api_key=AZURE_OPENAI_API_KEY, 
        azure_openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=DEPLOYMENT_NAME,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = await processor.process_excel_file(
        excel_file_path=file_buffer
    )

    print("[INFO] ----> Excel chunks created for data :- {}\n".format(data_path))
    print("[INFO] ----> Completed.")

    return chunks
