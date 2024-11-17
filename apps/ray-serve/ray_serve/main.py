import io
import logging
import time
import uuid

from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from ray import serve
from typing import Annotated
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, Response

app = FastAPI()
# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@serve.deployment(
    ray_actor_options={"num_gpus": 1, "num_cpus": 4},
    autoscaling_config={"min_replicas": 0, "max_replicas": 4},
)
@serve.ingress(app)
class BGRemover(object):
    def __init__(self):
        logger.info("Initializing BGRemover service.")
        self.model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded and moved to {self.device}.")

    @app.post(
        "/process_image",
        responses={
            200: {
                "description": "Processed image with background removed.",
                "content": {
                    "image/png": {
                        "example": "Binary image data (not displayed in OpenAPI)."
                    }
                },
            },
            400: {
                "description": "Invalid file uploaded.",
                "content": {
                    "application/json": {
                        "example": {"error": "Please upload an image file."}
                    }
                },
            },
            500: {
                "description": "Server error during image processing.",
                "content": {
                    "application/json": {
                        "example": {"error": "An error occurred during image processing."}
                    }
                },
            },
        },
        response_class=Response  # Ensures the endpoint returns a raw binary response.
    )
    async def process_image(self, file: Annotated[UploadFile, File(description="A file read as UploadFile")],
                            # fp16: bool = True # Force FP16 inference
                            ):
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        logger.info(f"[Request ID: {request_id}] Received request to process file: {file.filename}")

        try:
            start_time = time.time()

            # Check if the file is an image
            if file.content_type.split("/")[0] != "image":
                logger.warning(f"[Request ID: {request_id}] Uploaded file is not an image.")
                return JSONResponse(content={"error": "Please upload an image file."}, status_code=400)

            # Step 1: File validation
            step_start_time = time.time()
            if file.filename.split(".")[-1] not in ["jpg", "jpeg", "png"]:
                logger.warning(f"[Request ID: {request_id}] Invalid image file extension.")
                return JSONResponse(content={"error": "Please upload an image file with extension jpg, jpeg, or png."},
                                    status_code=400)
            step_elapsed_time = time.time() - step_start_time
            logger.info(f"[Request ID: {request_id}] File validation completed in {step_elapsed_time:.2f} seconds.")

            # Step 2: Load image
            step_start_time = time.time()
            image = Image.open(file.file)
            logger.info(f"[Request ID: {request_id}] Image loaded with size: {image.size}")
            step_elapsed_time = time.time() - step_start_time
            logger.info(f"[Request ID: {request_id}] Image loading completed in {step_elapsed_time:.2f} seconds.")

            # Step 3: Preprocess the image
            step_start_time = time.time()
            image_size = image.size
            transform_image = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_images = transform_image(image).unsqueeze(0).to(self.device)
            step_elapsed_time = time.time() - step_start_time
            logger.info(f"[Request ID: {request_id}] Image preprocessing completed in {step_elapsed_time:.2f} seconds.")

            # Step 4: Prediction
            step_start_time = time.time()
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=True):  # Enable FP16 inference
                    preds = self.model(input_images)[-1].sigmoid().cpu()
            step_elapsed_time = time.time() - step_start_time
            logger.info(f"[Request ID: {request_id}] Prediction completed in {step_elapsed_time:.2f} seconds.")

            # Step 5: Postprocess the prediction
            step_start_time = time.time()
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(image.size)
            image.putalpha(mask)
            step_elapsed_time = time.time() - step_start_time
            logger.info(f"[Request ID: {request_id}] Postprocessing completed in {step_elapsed_time:.2f} seconds.")

            # Step 6: Convert image to bytes
            step_start_time = time.time()
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")
            step_elapsed_time = time.time() - step_start_time
            logger.info(f"[Request ID: {request_id}] Image conversion completed in {step_elapsed_time:.2f} seconds.")

            total_elapsed_time = time.time() - start_time
            logger.info(f"[Request ID: {request_id}] Total processing time: {total_elapsed_time:.2f} seconds.")
            return Response(content=image_bytes.getvalue(), media_type="image/png")

        except Exception as e:
            logger.error(f"[Request ID: {request_id}] Error during image processing: {e}", exc_info=True)
            return JSONResponse(content={"error": "An error occurred during image processing."}, status_code=500)


app = BGRemover.bind(
    name="bgremover",
    version="1.0",
)
