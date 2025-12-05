# Use the official PySpark notebook image
FROM jupyter/pyspark-notebook:latest

# Switch to root to install system dependencies if needed
USER root

# (Optional) Install any system-level packages here
# RUN apt-get update && apt-get install -y curl

# Switch back to the default user
USER ${NB_UID}

# Copy requirements.txt into the container
COPY requirements.txt /tmp/

# Install Python dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r /tmp/requirements.txt