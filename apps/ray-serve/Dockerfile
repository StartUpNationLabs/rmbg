# Start with Ray's GPU base image
FROM rayproject/ray:2.31.0-py310-gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
  DEBIAN_FRONTEND=noninteractive \
  POETRY_VERSION=1.4.2 \
  POETRY_HOME="/opt/poetry" \
  POETRY_VENV="/opt/poetry-venv" \
  POETRY_CACHE_DIR="/opt/.cache"

# Install Poetry
RUN sudo apt update && sudo apt install curl -y --no-install-recommends && \
  curl -sSL https://install.python-poetry.org | sudo /home/ray/anaconda3/bin/python3 - && \
  sudo apt remove curl -y && sudo apt autoremove -y && sudo apt clean -y && \
  sudo rm -rf /var/lib/apt/lists/*

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files from the ray-serve app directory
COPY ./pyproject.toml ./poetry.lock ./

# Install dependencies using Poetry
RUN sudo poetry install
