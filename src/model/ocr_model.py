from pydantic import BaseModel, Field
from typing import List

class OCROutput(BaseModel):
    inference_result: List[tuple[str, float]] = Field(..., description="List of recognized text strings", examples=[[("NIK", 0.998), ("3333222", 0.8), ("Nama", 0.995), (": Johnny Andrean", 0.997)]])
    image_result: str = Field(
        ..., 
        description="Base64 encoded JPEG image for showing OCR results", 
        examples=["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."]
    )

class OCRResponse(BaseModel):
    filename: str = Field(..., description="Name of the uploaded file")
    outputs: OCROutput = Field(..., description="List of OCR results for each detected text region")
