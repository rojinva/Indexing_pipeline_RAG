#!/bin/bash
echo "Installing LibreOffice..."
if command -v apt-get &> /dev/null; then
     apt-get update
     apt-get install -y libreoffice-common || {
        echo "Failed to install LibreOffice."
        exit 1
    }
fi

