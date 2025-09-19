from pydantic import BaseModel, Field
from typing import List, Dict

class OCRJsonResult(BaseModel):
    res: List[str] = Field(..., description="List of recognized text strings", examples=[["PROVINSI", "NIK", "KABUPATEN", "KOTA"]])

class OCRImageResult(BaseModel):
    ocr_res_img: str = Field(
        ..., 
        description="Base64 encoded JPEG image for showing OCR results", 
        examples=["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."]
    )
class OCROutput(BaseModel):
    json_result: OCRJsonResult
    image_result: OCRImageResult

class OCRResponse(BaseModel):
    filename: str = Field(..., description="Name of the uploaded file")
    outputs: List[OCROutput] = Field(..., description="List of OCR results for each detected text region")
