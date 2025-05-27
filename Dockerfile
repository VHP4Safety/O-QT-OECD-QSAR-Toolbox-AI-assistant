# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml and the source code into the container
COPY pyproject.toml .
COPY src/ ./src/

# Install the project and its dependencies
# The --no-cache-dir flag is used to prevent pip from caching packages,
# which reduces the image size.
RUN pip install --no-cache-dir .

# Expose the port Streamlit runs on (default is 8501)
EXPOSE 8501

# Run the Streamlit application using the installed package's entry point
ENTRYPOINT ["oqt-assistant"]
