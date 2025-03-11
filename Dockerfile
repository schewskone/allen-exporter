# Use an official Python base image
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /src
COPY . /src

# Install system dependencies for AllenSDK and Jupyter
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -e /src/experanto \
    # && pip install --no-cache-dir -e /src/sensorium_2023 \ since there is no setup.py we can't do this
    && pip install -e /src/neuralpredictors


# Expose the default Jupyter Notebook port
EXPOSE 8888

# Set default command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
