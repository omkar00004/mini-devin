# Dockerfile
# Base image: official Python 3.11 minimal install
# slim = Debian with only Python, no extras (saves ~200MB vs full image)
FROM python:3.11-slim

# Install pytest into the image permanently.
# --no-cache-dir: don't store pip's download cache in the image layer.
# Keeps image size small — cache is useless inside a read-only container.
RUN pip install pytest --no-cache-dir