import platform
import subprocess
import shutil
import logging
from functools import lru_cache

@lru_cache(maxsize=1)
def install_poppler_utils():
    """
    Installs Poppler Utils dependencies if they are not already installed.

    This function checks the operating system and installs Poppler Utils using the appropriate method.
    On Linux, it uses the apt-get package manager. On Windows, it checks for the presence of pdftotext.exe.
    For unsupported operating systems, it raises a runtime error.
    Instruct the user to install it manually if not installed.

    Raises:
        RuntimeError: If Poppler Utils is not found on Windows or if the OS is unsupported.
    """
    system = platform.system()
    logging.info(f"Detected operating system: {system}")
    
    if system == "Linux":
        # Check if Poppler Utils is already installed
        result = subprocess.run(["which", "pdftoppm"], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info("Poppler Utils is already installed.")
        else:
            logging.info("Poppler Utils not found. Installing...")
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y", "poppler-utils"], check=True)
            logging.info("Poppler Utils installed successfully.")
    
    elif system == "Windows":
        # Check for pdftotext.exe and instruct the user if not found
        if not shutil.which("pdftoppm.exe"):
            raise RuntimeError(
                "Poppler Utils not found. Please install it manually "
                "(e.g., download from https://github.com/oschwartz10612/poppler-windows/releases) "
                "before running."
            )
        else:
            logging.info("Poppler Utils is already installed on Windows.")
    
    else:
        raise RuntimeError(f"Unsupported OS: {system}")