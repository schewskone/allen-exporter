# Use an official Python base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /src

# Install system dependencies for AllenSDK and Jupyter
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file into the container
COPY requirements.txt .

RUN git clone https://github.com/tomschewski/experanto.git /src/experanto

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir notebook \
    && pip install --no-cache-dir -e /src/experanto


# Expose the default Jupyter Notebook port
EXPOSE 8888

# Copy application code into the container
COPY ./allensdk_export /src

# Set default command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
