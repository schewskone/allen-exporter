FROM python:3.9-slim

# Set working directory in the container
WORKDIR /src
COPY . /src/allen_exporter
COPY requirements.txt /src

# Install system dependencies for AllenSDK and Jupyter
RUN apt-get update && \
        apt-get install -y --no-install-recommends git ffmpeg && \
        apt-get clean && rm -rf /var/liv/apt/lists/*

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -e /src/allen_exporter

EXPOSE 8888

# Set default command to run Jupyter Notebook
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
