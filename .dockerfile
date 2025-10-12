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

# Expose the port Streamlit uses (local default)
EXPOSE 8501
# For Hugging Face Spaces, use: EXPOSE 7860

# Set port environment variable (local default)
ENV PORT=8501
# For Hugging Face Spaces, use: ENV PORT=7860

# Streamlit config for local development
RUN mkdir -p ~/.streamlit && \
    echo "[server]\nport = 8501\nheadless = true\nallow_root = true\n" > ~/.streamlit/config.toml
# For Hugging Face Spaces, use: echo "[server]\nport = 7860\nheadless = true\nallow_root = true\nenableCORS = false\nenableXsrfProtection = false\n" > ~/.streamlit/config.toml

# Set Streamlit as the entrypoint (local default)
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# For Hugging Face Spaces, use: CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
