# Background Remover Service

This project implements a **Background Remover Service** using Ray Serve. It leverages PyTorch and Hugging Face Transformers to perform image segmentation and background removal. The service is exposed via a FastAPI interface for easy interaction.

## Features

- Removes backgrounds from uploaded images (JPEG, PNG).
- Supports inference on GPU or CPU.
- Autoscaling capabilities with Ray Serve.
- Logs performance metrics for debugging and optimization.

---

## Requirements

- Python 3.10 or later
- GPU-enabled system (optional but recommended for better performance)
- Ray Serve
- Required Python libraries (managed via Poetry)

---

## Installation and Setup

Follow these steps to set up the environment and run the service:

### 1. Clone the Repository

```bash
git clone https://github.com/StartUpNationLabs/rmbg
cd apps/ray-serve
```

### 2. Install Dependencies

#### Using Poetry

This project uses [Poetry](https://python-poetry.org/) for dependency management. Install Poetry if not already installed.
By defaut the project uses cuda. If you want to use cpu, you can remove the source in the pyproject.toml file for torch, torchvision and torchaudio.

```bash
Then, install the dependencies:

```bash
poetry install
```

## Running the Service

### 1. Start the Ray Cluster

Ensure Ray is running before starting the service:

```bash
ray start --head
```

### 2. Launch the Background Remover Service

Navigate to the `ray_serve` directory and run the service:

```bash
cd ray_serve
serve run main:app
```

### 3. Access the API

The FastAPI endpoints will be available at `http://127.0.0.1:8000`. Documentation for the API can be accessed at `http://127.0.0.1:8000/docs`.

#### Endpoint: `/process_image`

- **Method**: POST
- **Description**: Accepts an image file and returns a processed image with the background removed.
- **Request**:
  - Content-Type: `multipart/form-data`
  - Body: File field with the uploaded image.
- **Response**:
  - Success: A PNG image with the background removed.
  - Errors: JSON error messages with details about validation or processing issues.

---

## Configuration for Deployment

- **`serve_config.yaml`**: Manage Ray Serve deployment configurations such as autoscaling and replicas.

### Running using serve_config.yaml

```bash
serve run serve_config.yaml
```

This will download the version of the code set in the serve_config.yaml file and run the service.

## License

This project is licensed under [MIT License](LICENSE).

---

Happy Coding! 😊
