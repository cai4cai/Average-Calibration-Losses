FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and pip for the default Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common curl sudo \
        git python3-pip python3-venv python3-dev && \
    rm -rf /var/lib/apt/lists/*

# CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Create 'python' symlink for convenience
RUN ln -s /usr/bin/python3 /usr/local/bin/python

# Create non-root user
ARG USER
ARG USER_ID
ARG GROUP_ID
RUN groupadd --gid $GROUP_ID $USER && \
    useradd --uid $USER_ID --gid $GROUP_ID -m $USER -s /bin/bash && \
    echo "$USER ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USER && \
    chmod 0440 /etc/sudoers.d/$USER

CMD ["bash"]