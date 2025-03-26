#!/bin/bash
set -e
# URL for the Java tarball (adjust to your desired version/distribution)
DOWNLOAD_URL="https://download.java.net/java/GA/jdk11/13/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz"
# Desired target folder; adjust as needed
TARGET_DIR="/workspace/.devcontainer/jdk"
mkdir -p "$TARGET_DIR"
echo "Downloading Java from $DOWNLOAD_URL..."
curl -Lo java.tar.gz "$DOWNLOAD_URL"
echo "Extracting Java to $TARGET_DIR..."
tar -xzf java.tar.gz --strip-components=1 -C "$TARGET_DIR"
rm java.tar.gz
echo "Java downloaded and extracted to $TARGET_DIR"

// filepath: /Users/yinray/Documents/workspace-mac/llama_index/.devcontainer/download_java.sh
# ...existing code... 