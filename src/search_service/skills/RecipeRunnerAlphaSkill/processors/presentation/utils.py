import base64
import hashlib
import io
import logging
import os
import shutil
import subprocess
import tempfile
import platform
import urllib.parse
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from langchain_core.messages import HumanMessage
from pdf2image import convert_from_path
from pptx import Presentation
from PIL import Image

from .constants import FileExtensions
from .services import llm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_slide_to_base64(slide: Image.Image) -> str:
    """Convert a PIL Image slide to a base64 encoded string."""
    image_buffer = io.BytesIO()
    slide.save(image_buffer, format="JPEG")
    base64_image = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
    return base64_image


def convert_pptx_to_pdf_soffice(
    pptx_path: str,
    pdf_path: Optional[str] = None,
    directory: Optional[str] = None,
    show_hidden_slides: Optional[bool] = True,
) -> str:
    """Convert PPTX to PDF using LibreOffice soffice."""

    if not directory:
        directory = os.path.dirname(pptx_path)
    if not pdf_path:
        pdf_path = os.path.join(
            directory,
            os.path.basename(pptx_path).replace(
                FileExtensions.PPTX, FileExtensions.PDF
            ),
        )

    # Create a copy of the PPTX file to avoid modifying the original
    file_name = os.path.splitext(os.path.basename(pptx_path))[0]
    pptx_copy_path = os.path.join(directory, f"{file_name}_copy{FileExtensions.PPTX}")
    shutil.copy2(pptx_path, pptx_copy_path)

    # Ensure all hidden slides are visible
    if show_hidden_slides:
        try:
            presentation = Presentation(pptx_copy_path)
            for slide in presentation.slides:
                slide_element = slide.element
                if slide_element.get("show") == "0":
                    slide_element.set("show", "1")
            presentation.save(pptx_copy_path)
            logging.info("Successfully processed hidden slides")
        except Exception as e:
            logging.warning(f"Could not process hidden slides: {e}")

    # Convert the presentation to PDF
    command = [
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        directory,
        pptx_copy_path,
    ]
    logging.info("Converting PPTX to PDF using soffice")
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.stdout:
            logging.info(f"LibreOffice stdout: {result.stdout}")
        if result.stderr:
            logging.warning(f"LibreOffice stderr: {result.stderr}")
        if result.returncode != 0:
            logging.error(
                f"LibreOffice conversion failed with return code: {result.returncode}"
            )
            raise RuntimeError(f"LibreOffice conversion failed: {result.stderr}")
    except Exception as e:
        logging.error(f"Error running LibreOffice: {e}")
        raise RuntimeError(f"PDF conversion failed: {e}")

    # Change the name of the PDF file to match the original PPTX file name
    expected_pdf_path = os.path.join(
        directory,
        os.path.basename(pptx_copy_path).replace(
            FileExtensions.PPTX, FileExtensions.PDF
        ),
    )
    if expected_pdf_path != pdf_path:
        shutil.move(expected_pdf_path, pdf_path)
    return pdf_path


def convert_pptx_to_base64_image_data(
    pptx_path: str, dpi: Optional[int] = 200,
) -> List[str]:
    """
    Convert a PPTX file to a list of base64 encoded image bytes.

    Args:
        pptx_path (str): Path to the PPTX file.
        dpi (int): DPI for image conversion.

    Returns:
        List[str]: List of base64 encoded image bytes for each slide.
    """

    if not os.path.exists(pptx_path):
        raise FileNotFoundError(f"The PPTX file does not exist: {pptx_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Convert PPTX to PDF
        file_name = os.path.splitext(os.path.basename(pptx_path))[0]
        logging.info(f"Converting {file_name} to PDF format (*.pptx -> *.pdf)")

        pdf_path = None
        try:
            pdf_path = convert_pptx_to_pdf_soffice(
                pptx_path=pptx_path, directory=temp_dir
            )
            logging.info(f"Finished converting {file_name} to PDF: {pdf_path}")
        except Exception as e:
            logging.error(f"Error converting {file_name} to PDF: {e}")
            raise

        if not pdf_path or not os.path.exists(pdf_path):
            raise FileNotFoundError(f"No PDF file was found at {pdf_path}")

        # 2. Convert PDF to images and encode them in base64
        logging.info(f"Converting {file_name} to image format (PDF -> image).")
        slides: list[Image.Image] = convert_from_path(pdf_path, dpi=dpi)
        logging.info(
            f"Finished converting {file_name} to image ({len(slides)} images)."
        )

    logging.info("Converting images to base64 encoded strings.")
    try:
        with ThreadPoolExecutor() as executor:
            base64_image_data = list(executor.map(convert_slide_to_base64, slides))
        logging.info(
            f"Converted {len(base64_image_data)} images to base64 encoded strings."
        )
        return base64_image_data
    except Exception as e:
        logging.error(f"Error converting images to base64: {e}")
        raise RuntimeError(f"Image to base64 conversion failed: {e}")


async def extract_information_from_base64_image(base64_image: str, prompt: str) -> str:
    """Extract information from image by using LLM"""
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpg;base64,{base64_image}"},
            },
        ],
    )
    response = await llm.ainvoke([message])
    return response.content


def generate_hash(text: str):
    """Generate a SHA-256 hash of the given text."""
    hash_object = hashlib.sha256(text.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex


def remove_extension(file_name: str) -> str:
    """Remove file extension if it exists."""
    if "." in file_name:
        return file_name.rsplit(".", 1)[0]  # Remove extension
    return file_name


def parse_uri(blob_uri: str) -> Tuple[str, str, str]:
    """
    Parse Azure Blob Storage URL to extract container, directory, and file path components.

    Extracts the container name, top-level directory, and remaining path from an Azure Blob Storage URI.
    File extensions are automatically removed from the final path component.

    Args:
        blob_uri (str): Azure Blob Storage URL in the format:
            https://{storage_account}.blob.core.windows.net/{container_name}/{top_directory}/{remaining_path}

    Returns:
        Tuple[str, str, str]: A tuple containing:
            - container_name (str): The blob container name (e.g., 'knowledge-mining')
            - top_directory (str): The first directory level in the blob path
            - remaining_path (str): The remaining path with file extension removed

    Examples:
        parse_url_path("https://storage.blob.core.windows.net/container/top/sub/file.pdf")
        >>> ("container", "top", "sub/file")
    """

    parsed_uri = urllib.parse.urlparse(blob_uri)
    path = parsed_uri.path.lstrip("/")  # Remove leading slash
    parts = path.split("/", 2)  # Split into container_name, top_directory, and the rest
    if len(parts) < 3:
        raise ValueError(
            "URI does not contain enough parts to extract container name and directory."
        )

    container_name = parts[0]
    top_directory = parts[1]
    remaining_path = parts[2]

    if "/" in remaining_path:
        path_parts = remaining_path.split("/")
        file_name = path_parts[-1]
        file_name = remove_extension(file_name)
        path_parts[-1] = file_name
        remaining_path = "/".join(path_parts)
    else:
        remaining_path = remove_extension(remaining_path)

    return container_name, top_directory, remaining_path


def generate_base_media_path(
    blob_uri: str, path_modifier: Optional[List[str]] = None
) -> str:
    """
    Generate a media path based on the Azure Blob Storage URI and an optional path modifier.

    Creates a structured media path by extracting components from the blob URI and inserting
    custom path modifiers between the top directory and the remaining path.

    Args:
        blob_uri (str): The Azure Blob Storage URI in the format:
            https://{storage_account}.blob.core.windows.net/{container_name}/{top_directory}/{remaining_path}
        path_modifier (List[str], optional): A list of strings to insert into the path.
            Defaults to ["index-artifacts", "images"].

    Returns:
        str: The generated media path in the format:
            {container_name}/{top_directory}/{(joined) path_modifier}/{remaining_path}

    Examples:
        generate_base_media_path("https://storage.blob.core.windows.net/container/top/sub/file.pdf", ["index-artifacts", "images"])
        >>> "container/top/index-artifacts/images/sub/file"
    """

    # Set default path modifier if not provided
    if path_modifier is None:
        path_modifier = ["index-artifacts", "images"]

    # Validate path_modifier elements
    if not isinstance(path_modifier, list):
        raise TypeError("path_modifier must be a list")
    if not all(isinstance(item, str) for item in path_modifier):
        raise TypeError("All elements in path_modifier must be strings")

    # Filter out empty strings and normalize path separators
    path_modifier = [
        item.strip().replace("\\", "/") for item in path_modifier if item.strip()
    ]
    if not path_modifier:
        raise ValueError("path_modifier cannot be empty or contain only whitespace")
    path_modifier_joined = "/".join(path_modifier)

    # Construct the final media path
    try:
        container_name, top_directory, remaining_path = parse_uri(blob_uri)
    except ValueError as e:
        logging.error(f"Error parsing blob URI: {e}")
        raise ValueError(f"Invalid blob URI format: {e}")
    base_media_path = (
        f"{container_name}/{top_directory}/{path_modifier_joined}/{remaining_path}"
    )
    base_media_path = "/".join(part for part in base_media_path.split("/") if part)

    return base_media_path

@lru_cache(maxsize=1)
def install_libreoffice_dependencies():
    """
    Installs LibreOffice dependencies if they are not already installed.

    This function checks the operating system and installs LibreOffice using the appropriate method.
    On Linux, it uses the apt-get package manager. On Windows, it checks for the presence of soffice.exe.
    For unsupported operating systems, it raises a runtime error.
    Instruct the user to install it manually if not installed.

    Raises:
        RuntimeError: If LibreOffice is not found on Windows or if the OS is unsupported.
    """
    system = platform.system()
    
    if system == "Linux":
        # Only on Linux do we run apt-get to install LibreOffice
        result = subprocess.run(["which", "libreoffice"], capture_output=True, text=True)
        if result.returncode == 0:
            print("libreoffice is already installed.")
        else:
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y", "libreoffice"], check=True)
            print("libreoffice installed successfully.")
    
    elif system == "Windows":
        # On Windows, check for soffice.exe and raise an error if missing
        # If LibreOffice is not found, instruct the user to install it manually
        SOFFICE=r"C:\Program Files\LibreOffice\program\soffice.exe"
        path = SOFFICE
        if not shutil.which(path):
            raise RuntimeError(
                f"LibreOffice not found at {path}. "
                "Please install it (using GUI, winget, or choco) before running."
            )
    
    else:
        # Raise an error for unsupported operating systems
        raise RuntimeError(f"Unsupported OS: {system}")