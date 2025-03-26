FROM ubuntu:20.04

ENV PYTHONUNBUFFERED 1

# Copy the download script into the container
COPY download_java.sh /tmp/download_java.sh
RUN chmod +x /tmp/download_java.sh && /tmp/download_java.sh

# Set JAVA_HOME to the downloaded Java location
ENV JAVA_HOME=/workspace/.devcontainer/jdk
ENV PATH="$JAVA_HOME/bin:${PATH}"

# [Optional] If your requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>



