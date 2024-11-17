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
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="${POETRY_HOME}/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files from the ray-serve app directory
COPY ./pyproject.toml ./poetry.lock ./
COPY ray_serve ./ray_serve

# Install dependencies using Poetry
RUN poetry install

# Expose the port the app runs on
EXPOSE 8000

# Start the service (Ray is already configured in the base image)
CMD ["serve", "run", "ray_serve/main:app"]
