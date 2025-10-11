# Use official Python base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        wget \
        zlib1g-dev \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port Streamlit uses
EXPOSE 7860

# Hugging Face Spaces expects the app to run on port 7860
ENV PORT 7860

# Streamlit config for Hugging Face Spaces
RUN mkdir -p ~/.streamlit
RUN echo "[server]\nport = 7860\nheadless = true\nallow_root = true\n" > ~/.streamlit/config.toml

# Set Streamlit as the entrypoint
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]