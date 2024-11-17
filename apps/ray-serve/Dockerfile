# Start with Ray's GPU base image
FROM rayproject/ray:2.31.0-py310-gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
  DEBIAN_FRONTEND=noninteractive \
  POETRY_VERSION=1.8.4 \
  POETRY_HOME="/home/ray/poetry" \
  POETRY_VENV="/home/ray/poetry-venv" \
  POETRY_CACHE_DIR="/home/ray/.cache"

# Install Poetry
RUN sudo apt update && sudo apt install curl -y --no-install-recommends && \
  curl -sSL https://install.python-poetry.org | python3 - && \
  sudo apt remove curl -y && sudo apt autoremove -y && sudo apt clean -y && \
  sudo rm -rf /var/lib/apt/lists/*

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files from the ray-serve app directory
COPY ./pyproject.toml ./poetry.lock ./

# Install dependencies using Poetry
RUN poetry install
