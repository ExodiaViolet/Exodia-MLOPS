ARG BASE_IMAGE=python:3.9-slim
FROM $BASE_IMAGE as runtime-environment

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install rustup and latest stable Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# Add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

WORKDIR /home/kedro_docker
USER kedro_docker

FROM runtime-environment

ARG KEDRO_UID=999
ARG KEDRO_GID=0

# Copy all files with correct ownership
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

EXPOSE 8888

# Switch back to root to run chmod
USER root

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Switch to kedro user to run the script (optional but recommended)
USER kedro_docker

CMD ["/app/start.sh"]
