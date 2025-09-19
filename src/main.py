import tempfile
import time
import aiofiles
import os
from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import Union
from service.ocr_service import handle_ocr_inference
from model.ocr_model import OCRResponse, OCRImageResult, OCRJsonResult, OCROutput
import asyncio

app = FastAPI(title="OCR API", description="API for OCR processing", version="1.0.0", root_path="/api/v1")
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)


@app.post(
    "/run_ocr/",
    description="Run OCR on an uploaded file and return the results via images and json",
    summary="Run OCR on an uploaded file"   ,
    response_description="OCR results including processed images and JSON data",
    response_model=OCRResponse,
)
async def run_ocr(image_file: UploadFile, response: Response) -> Union[OCRResponse, dict]:
    try:
        start_time = time.perf_counter()
        
        # Create a temporary file path
        fd, temp_path = tempfile.mkstemp(suffix=f"_{image_file.filename}")
        os.close(fd)

        contents = await image_file.read()
        async with aiofiles.open(temp_path, "wb") as out_file:
            await out_file.write(contents)

        outputs = None
        try:
            outputs = await asyncio.to_thread(handle_ocr_inference, temp_path)
        finally:
            os.remove(temp_path)

        elapsed_time = time.perf_counter() - start_time
        response.headers["X-Processing-Time"] = f"{elapsed_time:.3f}s"

        return {
            "filename": image_file.filename,
            "outputs": outputs
        }
    except Exception as e:
        response.status_code = 500
        print(f"Error processing file: {e}")
        return {
            "filename": image_file.filename if 'image_file' in locals() else "unknown",
            "outputs": [],
        }
